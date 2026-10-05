import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator


def _build_prompt(state: MERMAIDState) -> str:
    """
    Builds the correct text prompt for the Decision Agent based on query_mode.
    """
    labels       = ", ".join(state["candidate_emotions"])
    caption      = state.get("caption", "")
    query_mode   = state.get("query_mode", "qinit")
    prediction   = state.get("current_prediction", "")

    if query_mode == "qinit":
        text = (
            f"This image must be classified into one of these human emotions: {labels}.\n"
            "Based on psychological and visual features such as facial expression, body posture, "
            "background context, lighting, and colour tones, please analyse the emotional state "
            "represented.\n"
            f'Image Description: "{caption}"\n'
            "Think carefully and choose only one emotion word from the list above."
        )
    else:
        if query_mode == "qtext":
            feedback = state.get("textual_feedback") or {}
        else:  # "qvis"
            feedback = state.get("visual_feedback") or {}

        suggestion  = feedback.get("suggestion", prediction)
        fb_text     = feedback.get("justification", "")

        text = (
            "Image Context for Current Task:\n"
            f'Image Caption: "{caption}"\n'
            "IMPORTANT GUIDANCE FOR REFINEMENT:\n"
            f'A previous reflection on the emotion "{prediction}" suggested the new emotion '
            f'might be "{suggestion}" based on the following feedback: "{fb_text}". '
            "Please carefully consider this feedback.\n"
            f"Your task is to choose the best emotion label for the image from the following list: "
            f"{labels}. Output only the chosen emotion word."
        )

    return text


from qwen_vl_utils import process_vision_info

def run_decision_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    mode = state.get("query_mode", "qinit")
    print(f"--- [Agent: Decision | mode={mode}] Iteration {state.get('iteration', 0)} ---")

    # 1. Ensure MLLM is on GPU
    memory_manager.load_mllm()
    model     = memory_manager.mllm
    processor = memory_manager.processor

    # 2. Load image path
    image_path = state["image_path"]

    # 3. Build the mode-appropriate prompt
    prompt_text = _build_prompt(state)

    # 4. Prepare Qwen inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # 5. Map raw output to a valid candidate (case-insensitive, fallback to first)
    prediction = next(
        (e for e in state["candidate_emotions"] if e.lower() in raw_output.lower()),
        state["candidate_emotions"][0],
    )

    print(f"    -> Predicted: '{prediction}'  (raw: '{raw_output}')")

    return {"current_prediction": prediction}

