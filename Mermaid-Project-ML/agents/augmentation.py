import os
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.qformer import resize_image_to_512

# Rich affective prompt dictionary mapping each emotion to lighting, color, and mood keywords (for use_qformer=False)
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
    """
<<<<<<< HEAD
    Generates reference images for candidate emotions using InstructPix2Pix with LCM.
    Caches reference images per input image so repeated reflection loops do not re-run diffusion.
    """
    img_stem = os.path.splitext(os.path.basename(state['image_path']))[0]
    cache_dir = os.path.join("generated_refs", img_stem)
    os.makedirs(cache_dir, exist_ok=True)
    
    existing_refs = state.get("reference_images", {})
    emotions_needed = [
        e for e in state["candidate_emotions"]
        if e not in existing_refs or not os.path.exists(existing_refs[e])
    ]
    
    if not emotions_needed:
        print(f"--- [Agent: Augmentation] Reusing all cached references for {img_stem} ---")
        return {"reference_images": existing_refs}
        
    print(f"--- [Agent: Augmentation] Generating References ({len(emotions_needed)} needed) ---")
    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline
    original_image = Image.open(state['image_path']).convert("RGB").resize((512, 512))
    
    generated_refs = dict(existing_refs)
    
    for emotion in emotions_needed:
        save_path = os.path.join(cache_dir, f"ref_{emotion.lower().replace(' ', '_')}.jpg")
        if os.path.exists(save_path):
            generated_refs[emotion] = save_path
            print(f"    - Reusing cached ref for '{emotion}' at {save_path}")
            continue

        prompt = f"make them look {emotion}, express {emotion}"
        with torch.no_grad():
            result_image = pipeline(
                prompt=prompt,
                image=original_image,
                num_inference_steps=4,
                guidance_scale=1.5,
                image_guidance_scale=1.5,
                output_type="pil"
            ).images[0]
            
        result_image.save(save_path)
        generated_refs[emotion] = save_path
        print(f"    -> Generated ref for '{emotion}' at {save_path}")
        
=======
    Generates reference images for ALL candidate emotions using Q-Former conditioned InstructPix2Pix with LCM LoRA
    (or rich affective prompts when use_qformer=False).
    """
    # ── Fast path (A3): reuse already-generated refs for this image ──────────
    existing_refs = state.get("reference_images", {})
    if existing_refs and all(os.path.exists(p) for p in existing_refs.values()):
        print("--- [Agent: Augmentation] Reusing existing reference images (skipping diffusion) ---")
        return {"reference_images": existing_refs}

    use_qformer = state.get("use_qformer", True)

    # ── A6 Fail-Fast Check: Validate Q-Former checkpoint existence ──────────
    if use_qformer:
        checkpoint_path = getattr(memory_manager, "qformer_checkpoint_path", "checkpoint/Q-Former.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"[A6 Fail-Fast] Q-Former checkpoint not found at '{checkpoint_path}'. "
                "Ensure weights exist or run scripts/download_qformer.py before running with use_qformer=True."
            )

    mode_label = "Q-Former + LCM" if use_qformer else "Affective Prompt + LCM"
    print(f"--- [Agent: Augmentation] Generating Emotion References ({mode_label}) ---")

    # 1. Ensure Diffusion Model is on GPU (offloading the MLLM via B10)
    memory_manager.load_diffusion()
    pipeline = memory_manager.diffusion_pipeline

    # 2. Load original image and resize to 512 short-side per A5 (multiples of 8)
    original_image = Image.open(state['image_path']).convert("RGB")
    input_image = resize_image_to_512(original_image)

    generated_refs = {}
    os.makedirs("generated_refs", exist_ok=True)

    # 3. LCM generation hyperparameters (fixed per A4 & B11: steps=6, guidance=1.8, image_guidance=1.5)
    num_steps = 6
    guidance = 1.8
    img_guidance = 1.5

    # 4. Extract CLIP image features ONCE per input image in torch.float16 if using Q-Former
    img_embeds = None
    if use_qformer:
        img_embeds = memory_manager.extract_clip_image_embeds(input_image)

    for emotion in state['candidate_emotions']:
        with torch.no_grad():
            if use_qformer:
                # Run the 12-layer EmoEdit Q-Former for each candidate emotion
                prompt_embeds = memory_manager.get_emotion_prompt_embeds(
                    emotion=emotion,
                    img_embeds=img_embeds,
                )

                # CRITICAL SHAPE GUARD (A4 & B11):
                # Ensure prompt_embeds and negative_prompt_embeds passed into
                # StableDiffusionInstructPix2PixPipeline have the exact same sequence length
                # and matching torch.float16 dtype so torch.cat inside diffusers never throws shape mismatch.
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

                # Exact sequence length match
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

                result_image = pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image=input_image,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    image_guidance_scale=img_guidance,
                    output_type="pil",
                ).images[0]
            else:
                # Rich affective prompt dictionary mapping each emotion to lighting, color, mood keywords
                prompt_text = get_affective_prompt(emotion)
                result_image = pipeline(
                    prompt=prompt_text,
                    image=input_image,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    image_guidance_scale=img_guidance,
                    output_type="pil",
                ).images[0]

        # A3: Cache the 8 generated reference images to disk at 256x256
        result_image = result_image.resize((256, 256), Image.LANCZOS)
        save_path = f"generated_refs/ref_{emotion.lower().replace(' ', '_')}.jpg"
        result_image.save(save_path)
        generated_refs[emotion] = save_path
        print(f"    -> [{mode_label}] Generated ref for '{emotion}' at {save_path}")

>>>>>>> 3ad1e63d855509b8bde5996995563a4d6c40f22f
    memory_manager._offload_diffusion()
    return {"reference_images": generated_refs}
