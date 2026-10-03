import json
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.schemas import ReflectionOutput

def generate_caption(image_path: str, memory_manager: MemoryOrchestrator) -> str:
    """Helper to generate a caption for an image."""
    model = memory_manager.mllm
    processor = memory_manager.processor
    
    image = Image.open(image_path).convert("RGB")
    prompt = "[INST] <image>\nDescribe this image in terms of emotional content. Include expressions, body language, scene context, dominant colours, and any object that influences mood. [/INST]"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=150)
        
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

def run_visual_reflection(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Compares the original image against the generated reference images to provide
    visual feedback on the prediction.
    
    Input:
        - state: The current LangGraph state containing 'image_path', 'reference_images', 'current_prediction', and 'candidate_emotions'.
        - memory_manager: Memory manager providing access to the loaded LLaVA-NeXT model.
        
    Output:
        - dict: A dictionary containing 'visual_feedback' with keys: status, suggestion, justification.
    """
    print("--- [Agent: Visual Reflection] ---")
    
    # 1. Ensure MLLM is on GPU (This will offload the Diffusion model!)
    memory_manager.load_mllm()
    
    model = memory_manager.mllm
    processor = memory_manager.processor
    
    # 2. Generate captions for all reference images to bypass LLaVA's single-image constraint
    ref_contexts = []
    print("    -> Generating captions for reference images...")
    for emotion, ref_img_path in state.get('reference_images', {}).items():
        ref_caption = generate_caption(ref_img_path, memory_manager)
        ref_contexts.append(f'Label: "{emotion}" | Description: "{ref_caption}"')
        
    ref_context_str = "\n".join(ref_contexts)
    
    # 3. Construct the prompt matching the paper's design
    instruction = (
        f"You are a visual reflection agent tasked with assessing the emotion of a given query image.\n"
        f"The predicted emotion is: \"{state['current_prediction']}\". Use the caption: \"{state['caption']}\" and compare it with the reference images.\n"
        f"You must assess emotional fit based on: - facial/body expression similarity - background and compositional coherence - lighting, atmosphere, and colour-based affective cues\n"
        f"Reference labels: {', '.join(state['candidate_emotions'])}\n"
        f"Reference Images Context:\n{ref_context_str}\n\n"
        f"Determine whether the predicted emotion is accurate. If not, suggest a better one and explain why.\n"
        f'Example (if correct):\n'
        f'"status: good_enough, suggestion: {state["current_prediction"]}, feedback: Based on the caption, the current label seems appropriate."\n'
        f'Example (if incorrect):\n'
        f'"status: needs_revision, suggestion: Anger, feedback: The caption mentions "shouting" and "aggressive posture", suggesting Anger is more fitting."\n\n'
        f"Output ONLY a valid JSON object matching this schema:\n"
        "{{\n"
        '  "status": "good_enough" or "needs_revision",\n'
        '  "suggestion": "<label>",\n'
        '  "feedback": "<explanation>"\n'
        "}}\n"
        "Do not include any other text."
    )
    
    prompt = f"[INST] {instruction}\n<image> [/INST]"
    original_image = Image.open(state['image_path']).convert("RGB")
    
    # 4. Call MLLM with a single image
    inputs = processor(text=prompt, images=original_image, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=150)
        
    input_len = inputs["input_ids"].shape[1]
    raw_output = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    
    # 5. Parse JSON
    try:
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()
            
        # The LLM frequently outputs "good\_enough" which is invalid JSON.
        raw_output = raw_output.replace('\\_', '_')
            
        # Validate against schema (mapping 'feedback' to 'justification' happens via model validation if we passed it direct, 
        # but the schema expects 'justification'. Let's load JSON then validate)
        raw_dict = json.loads(raw_output)
        
        # The prompt for visual_reflection asks for "feedback", but schema needs "justification"
        if "feedback" in raw_dict and "justification" not in raw_dict:
            raw_dict["justification"] = raw_dict.pop("feedback")
            
        parsed = ReflectionOutput.model_validate(raw_dict)
        
        # Ensure suggestion is valid
        suggestion = parsed.suggestion
        matched = next((e for e in state['candidate_emotions'] if e.lower() == suggestion.lower()), None)
        suggestion = matched if matched else state['current_prediction']
        
        feedback = {
            "status": parsed.status,
            "suggestion": suggestion,
            "justification": parsed.justification
        }
            
    except Exception as e:
        print(f"    [Warning] JSON parse failed: {e} \n Raw Output: {raw_output}")
        feedback = {
            "status": "needs_revision",
            "suggestion": state['current_prediction'],
            "justification": "Failed to parse MLLM output."
        }
        
    print(f"    -> Feedback Status: {feedback['status']}")
    
    return {
        "visual_feedback": feedback
    }
