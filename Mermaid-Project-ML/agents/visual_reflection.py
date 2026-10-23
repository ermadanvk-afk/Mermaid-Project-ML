import json
import re
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.schemas import ReflectionOutput
from agents.caption_agent import run_llava_generation

def run_visual_reflection(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Compares original image against generated references using LLaVA-NeXT.
    Bypasses single-image input limit by first captioning each reference image,
    then doing multimodal reflection with query image + reference captions.
    """
    print("--- [Agent: Visual Reflection] ---")
    
    # 1. Caption each generated reference image
    ref_contexts = []
    print("    -> Describing generated emotion reference images...")
    for emotion, ref_img_path in state.get('reference_images', {}).items():
        ref_prompt = "Describe this image in terms of emotional content. Include expressions, body language, scene context, dominant colours, and any object that influences mood."
        ref_caption = run_llava_generation(ref_prompt, ref_img_path, memory_manager, max_new_tokens=150)
        ref_contexts.append(f'Label: "{emotion}" | Description: "{ref_caption}"')
        
    ref_context_str = "\n".join(ref_contexts)
    
    # 2. Construct visual reflection prompt
    instruction = (
        "You are a visual reflection agent tasked with assessing the emotion of a given query image.\n"
        f"The predicted emotion is: \"{state['current_prediction']}\". Use the caption: \"{state['caption']}\" and compare it with the reference images.\n"
        "You must assess emotional fit based on: - facial/body expression similarity - background and compositional coherence - lighting, atmosphere, and colour-based affective cues\n"
        f"Reference labels: {', '.join(state['candidate_emotions'])}\n"
        f"Reference Images Context:\n{ref_context_str}\n\n"
        "Determine whether the predicted emotion is accurate. If not, suggest a better one and explain why.\n"
        "Output ONLY a valid JSON object matching this schema:\n"
        "{{\n"
        '  "status": "good_enough" or "needs_revision",\n'
        '  "suggestion": "<label>",\n'
        '  "justification": "<explanation>"\n'
        "}}\n"
        "Do not include any other text."
    )
    
    # 3. Multimodal reflection query
    raw_output = run_llava_generation(instruction, state['image_path'], memory_manager, max_new_tokens=300)
    
    # 4. Parse JSON
    cleaned = raw_output.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    cleaned = cleaned.replace('\\_', '_')

    try:
        raw_dict = json.loads(cleaned)
        if "feedback" in raw_dict and "justification" not in raw_dict:
            raw_dict["justification"] = raw_dict.pop("feedback")
        parsed = ReflectionOutput.model_validate(raw_dict)
        matched = next((e for e in state['candidate_emotions'] if e.lower() == parsed.suggestion.lower()), None)
        feedback = {
            "status": parsed.status,
            "suggestion": matched if matched else state['current_prediction'],
            "justification": parsed.justification
        }
    except Exception:
        # Regex fallback
        status_m = re.search(r'"status"\s*:\s*"(good_enough|needs_revision)"', cleaned, re.IGNORECASE)
        sugg_m = re.search(r'"suggestion"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
        just_m = re.search(r'"(justification|feedback)"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
        if status_m:
            raw_sugg = sugg_m.group(1) if sugg_m else state['current_prediction']
            matched = next((e for e in state['candidate_emotions'] if e.lower() == raw_sugg.lower()), None)
            feedback = {
                "status": status_m.group(1).lower(),
                "suggestion": matched if matched else state['current_prediction'],
                "justification": just_m.group(2) if just_m else "Visual regex fallback."
            }
        else:
            feedback = {
                "status": "needs_revision",
                "suggestion": state['current_prediction'],
                "justification": "Visual parse fallback."
            }
        
    print(f"    -> Feedback Status: {feedback['status']} | Suggestion: {feedback['suggestion']}")
    return {"visual_feedback": feedback}
