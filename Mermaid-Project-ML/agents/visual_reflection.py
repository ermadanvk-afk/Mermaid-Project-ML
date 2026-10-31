import os
import json
import re
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.schemas import ReflectionOutput


def generate_caption(image_path: str, memory_manager: MemoryOrchestrator) -> str:
    """
    Helper to generate a concise emotional caption for an image using LLaVA 1.5 7B.
    Delegates to memory_manager.generate_mllm_response() so the proper
    LLaVA conversation chat-template is applied automatically.
    Capped at 512x512 resolution (B7 fix).
    """
    raw_image = Image.open(image_path).convert("RGB")
    raw_image.thumbnail((512, 512), Image.LANCZOS)
    prompt_text = (
        "Describe this image in terms of emotional content. "
        "Include expressions, body language, scene context, dominant colours, "
        "and any object that influences mood."
    )
    caption = memory_manager.generate_mllm_response(
        image=raw_image,
        prompt_text=prompt_text,
        max_new_tokens=60,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )
    return caption


def _parse_visual_reflection_output(raw_output: str, current_prediction: str, candidate_emotions: list) -> dict:
    """
    Safely parses JSON from MLLM output, handling markdown, conversational preambles,
    non-dictionary JSON values (e.g. raw numbers), and providing resilient fallbacks.
    """
    text = raw_output.replace('\\_', '_').strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Extract JSON object { ... } via regex
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    raw_dict = None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            raw_dict = loaded
        else:
            print(f"    [Warning] Parsed JSON is not a dictionary (type {type(loaded).__name__}): {loaded}")
    except Exception as e:
        print(f"    [Warning] JSON decode failed: {e}. Raw: {raw_output}")

    if raw_dict is not None:
        try:
            # Map legacy 'feedback' key to 'justification'
            if "feedback" in raw_dict and "justification" not in raw_dict:
                raw_dict["justification"] = raw_dict.pop("feedback")

            parsed = ReflectionOutput.model_validate(raw_dict)
            suggestion = parsed.suggestion
            matched = next((e for e in candidate_emotions if e.lower() == suggestion.lower()), None)
            suggestion = matched if matched else current_prediction

            return {
                "status": parsed.status,
                "suggestion": suggestion,
                "justification": parsed.justification,
            }
        except Exception as val_err:
            print(f"    [Warning] Schema validation failed: {val_err}. Dict: {raw_dict}")

    # Resilient heuristic fallback if JSON structure is missing
    lower = raw_output.lower()
    if "good_enough" in lower or "good enough" in lower:
        status = "good_enough"
        suggestion = current_prediction
    elif "needs_revision" in lower or "needs revision" in lower:
        status = "needs_revision"
        matched = next((e for e in candidate_emotions if e.lower() != current_prediction.lower() and e.lower() in lower), None)
        suggestion = matched if matched else current_prediction
    else:
        status = "needs_revision"
        suggestion = current_prediction

    justification = raw_output[:120].strip() if raw_output else "Parsed via heuristic fallback."
    return {
        "status": status,
        "suggestion": suggestion,
        "justification": justification,
    }


def run_visual_reflection(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Compares the original image against the generated reference images to provide
    visual feedback on the prediction using LLaVA 1.5 7B.

    Features B6 disk/state reference caption caching (0 MLLM calls on cache hit) and B7 resolution capping.
    """
    print("--- [Agent: Visual Reflection | LLaVA 1.5 7B] ---")
    
    # 1. Ensure LLaVA 1.5 7B is loaded on GPU
    memory_manager.load_mllm()

    # 2. Reference caption caching (B6 fix): check state and on-disk cache
    image_path = state['image_path']
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    cache_dir = "generated_refs"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{image_id}_ref_captions.json")

    ref_images = state.get('reference_images', {})
    cached_captions = dict(state.get("reference_captions") or {})
    loaded_from_cache = False

    # Check 1: In state
    if cached_captions and len(cached_captions) == len(ref_images) and all(e in cached_captions for e in ref_images.keys()):
        print(f"--- [Agent: Visual Reflection] Reusing reference captions from state for {image_id} (0 MLLM calls) ---")
        loaded_from_cache = True
    # Check 2: On disk
    elif os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                disk_captions = json.load(f)
            if isinstance(disk_captions, dict) and len(disk_captions) == len(ref_images) and all(e in disk_captions for e in ref_images.keys()):
                cached_captions = disk_captions
                print(f"--- [Agent: Visual Reflection] Loaded reference captions from disk cache {cache_file} (0 MLLM calls) ---")
                loaded_from_cache = True
        except Exception as e:
            print(f"    [Warning] Failed to load disk caption cache {cache_file}: {e}")

    # Generate reference captions if not cached
    if not loaded_from_cache:
        print("    -> Generating captions for reference images...")
        for emotion, ref_img_path in ref_images.items():
            if emotion not in cached_captions:
                ref_caption = generate_caption(ref_img_path, memory_manager)
                cached_captions[emotion] = ref_caption
                print(f"       Generated caption for '{emotion}'")

        # Save to disk cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_captions, f, indent=2)
            print(f"    -> Cached reference captions to {cache_file}")
        except Exception as e:
            print(f"    [Warning] Failed to write cache file {cache_file}: {e}")

    ref_contexts = []
    for emotion, ref_img_path in ref_images.items():
        ref_caption = cached_captions.get(emotion, "")
        short_caption = ref_caption.split(".")[0].strip() if "." in ref_caption else ref_caption[:120].strip()
        if not short_caption:
            short_caption = ref_caption[:120].strip()
        ref_contexts.append(f'- {emotion}: {short_caption}')

    ref_context_str = "\n".join(ref_contexts)

    # 3. Construct the visual-reflection prompt for LLaVA-NeXT 7B
    instruction = (
        f"You are a visual reflection agent tasked with assessing the emotion of a given query image.\n"
        f"The predicted emotion is: \"{state['current_prediction']}\". "
        f"Use the caption: \"{state['caption']}\" and compare it with the reference images.\n"
        f"Assess emotional fit based on:\n"
        f"  - Facial/body expression similarity\n"
        f"  - Background and compositional coherence\n"
        f"  - Lighting, atmosphere, and colour-based affective cues\n"
        f"Reference labels: {', '.join(state['candidate_emotions'])}\n"
        f"Reference Images Context:\n{ref_context_str}\n\n"
        f"Determine whether the predicted emotion is accurate. If not, suggest a better one and explain why.\n"
        f"Example (if correct):\n"
        f'"status: good_enough, suggestion: {state["current_prediction"]}, '
        f'feedback: Based on the caption, the current label seems appropriate."\n'
        f"Example (if incorrect):\n"
        f'"status: needs_revision, suggestion: Anger, '
        f'feedback: The caption mentions shouting and aggressive posture, suggesting Anger is more fitting."\n\n'
        f"Output ONLY a valid JSON object matching this schema:\n"
        "{{\n"
        '  "status": "good_enough" or "needs_revision",\n'
        f'  "suggestion": "{state["current_prediction"]}",\n'
        '  "justification": "<brief explanation>"\n'
        "}}\n"
        "Do not include any other text."
    )

    # 4. Cap query image resolution (B7 fix: max dim 512x512, max_pixels <= 262144)
    raw_image = Image.open(image_path).convert("RGB")
    raw_image.thumbnail((512, 512), Image.LANCZOS)

    # 5. Run LLaVA 1.5 7B via chat-template pipeline
    raw_output = memory_manager.generate_mllm_response(
        image=raw_image,
        prompt_text=instruction,
        max_new_tokens=150,
        repetition_penalty=1.15,
        no_repeat_ngram_size=0,
    )
    
    # 6. Parse JSON with safe fallback
    feedback = _parse_visual_reflection_output(
        raw_output=raw_output,
        current_prediction=state['current_prediction'],
        candidate_emotions=state['candidate_emotions'],
    )
        
    print(f"    -> Feedback Status: {feedback['status']}")
    
    return {
        "visual_feedback": feedback,
        "reference_captions": cached_captions,
    }
