import os
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.qformer import resize_image_to_512

AFFECTIVE_PROMPT_MAP = {
    "amusement": "A lighthearted and humorous scene with warm, playful lighting, cheerful vibrant colors, smiling expressions, and an entertaining mood.",
    "anger": "An intense, hostile scene with harsh contrasting shadows, aggressive sharp lighting, stormy reddish tones, and enraged furrowed expressions.",
    "awe": "A breathtaking and majestic view with grand scale, ethereal glowing lighting, deep atmospheric colors, and an overwhelming sense of wonder.",
    "contentment": "A serene, peaceful scene with soft warm golden-hour lighting, balanced pastel colors, relaxed posture, and tranquil comfort.",
    "disgust": "A repulsive, uncomfortable scene with sickly greenish-yellow tones, dirty textures, gloomy lighting, and revolted recoiling expressions.",
    "excitement": "A dynamic, high-energy scene with vibrant saturated colors, vivid luminous lighting, enthusiastic gestures, and an exhilarating mood.",
    "fear": "A chilling and terrifying scene with ominous dark shadows, cold desaturated blue tones, dim flickering light, and wide-eyed dread.",
    "sadness": "A somber, melancholic scene with muted dreary grey tones, soft diffused overcast lighting, downward gaze, and a lonely sorrowful mood.",
    "surprise": "A startling and unexpected moment with sudden dramatic focal illumination, striking vivid contrast, and wide open-mouthed astonishment.",
    "joy": "A radiant, jubilant scene with bright warm sunlight, vibrant lively colors, celebratory atmosphere, and ecstatic joyful expressions.",
}


def get_affective_prompt(emotion: str) -> str:
    """Returns a rich affective descriptive prompt for an emotion label."""
    key = emotion.lower().strip()
    return AFFECTIVE_PROMPT_MAP.get(
        key,
        f"A vivid affective visual conveying {emotion} with emotive lighting, expressive composition, and evocative color tones."
    )


def run_augmentation_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    # Reuse existing references when available
    existing_refs = state.get("reference_images", {})
    if existing_refs and all(os.path.exists(p) for p in existing_refs.values()):
        print("--- [Agent: Augmentation] Reusing existing reference images (skipping diffusion) ---")
        return {"reference_images": existing_refs}

    use_qformer = state.get("use_qformer", True)

    # Validate the Q-Former checkpoint
    if use_qformer:
        checkpoint_path = getattr(memory_manager, "qformer_checkpoint_path", "checkpoint/Q-Former.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"[A6 Fail-Fast] Q-Former checkpoint not found at '{checkpoint_path}'. "
                "Ensure weights exist or run scripts/download_qformer.py before running with use_qformer=True."
            )

    mode_label = "Q-Former + LCM" if use_qformer else "Affective Prompt + LCM"
    print(f"--- [Agent: Augmentation] Generating Emotion References ({mode_label}) ---")

    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline

    # Load and resize the input image
    original_image = Image.open(state['image_path']).convert("RGB")
    input_image = resize_image_to_512(original_image)

    generated_refs = {}
    os.makedirs("generated_refs", exist_ok=True)

    num_steps = 8
    guidance = 4.5
    img_guidance = 1.5

    img_embeds = None
    if use_qformer:
        img_embeds = memory_manager.extract_clip_image_embeds(input_image)

    seed = 42

    for emotion in state['candidate_emotions']:
        pipe_dev = getattr(pipeline, "device", None)
        if isinstance(pipe_dev, (torch.device, str)):
            generator = torch.Generator(device=pipe_dev).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            if use_qformer:
                prompt_embeds = memory_manager.get_emotion_prompt_embeds(
                    emotion=emotion,
                    img_embeds=img_embeds,
                )

                if prompt_embeds.shape[1] == 76:
                    padding = torch.zeros(
                        (prompt_embeds.shape[0], 1, prompt_embeds.shape[2]),
                        dtype=prompt_embeds.dtype,
                        device=prompt_embeds.device,
                    )
                    prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)

                uncond_tokens = pipeline.tokenizer(
                    [""],
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                negative_prompt_embeds = pipeline.text_encoder(
                    uncond_tokens.input_ids.to(pipeline.device)
                )[0].to(dtype=torch.float16, device=pipeline.device)

                # Match embedding lengths
                if negative_prompt_embeds.shape[1] > prompt_embeds.shape[1]:
                    negative_prompt_embeds = negative_prompt_embeds[:, :prompt_embeds.shape[1], :]
                elif negative_prompt_embeds.shape[1] < prompt_embeds.shape[1]:
                    pad_len = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                    pad = torch.zeros(
                        (negative_prompt_embeds.shape[0], pad_len, negative_prompt_embeds.shape[2]),
                        dtype=negative_prompt_embeds.dtype,
                        device=negative_prompt_embeds.device,
                    )
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)

                prompt_embeds = prompt_embeds.to(dtype=torch.float16, device=pipeline.device)
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.float16, device=pipeline.device)

                with torch.autocast("cuda"):
                    result_image = pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        image=input_image,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        image_guidance_scale=img_guidance,
                        generator=generator,
                        output_type="pil",
                    ).images[0]
            else:
                prompt_text = get_affective_prompt(emotion)
                with torch.autocast("cuda"):
                    result_image = pipeline(
                        prompt=prompt_text,
                        image=input_image,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance,
                        image_guidance_scale=img_guidance,
                        generator=generator,
                        output_type="pil",
                    ).images[0]

        # Save the generated reference image
        result_image = result_image.resize((256, 256), Image.LANCZOS)
        save_path = f"generated_refs/ref_{emotion.lower().replace(' ', '_')}.jpg"
        result_image.save(save_path)
        generated_refs[emotion] = save_path
        print(f"    -> [{mode_label}] Generated ref for '{emotion}' at {save_path}")

    memory_manager._offload_diffusion()
    return {"reference_images": generated_refs}
