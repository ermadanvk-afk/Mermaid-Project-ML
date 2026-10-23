import os
import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from agents.caption_agent import run_llava_generation

def _build_prompt(state: MERMAIDState) -> str:
    """
    Builds the correct text prompt for the Decision Agent based on query_mode.
    """
    labels     = ", ".join(state["candidate_emotions"])
    caption    = state.get("caption", "")
    query_mode = state.get("query_mode", "qinit")
    prediction = state.get("current_prediction", "")

    if query_mode == "qinit":
        return (
            f"This image must be classified into one of these human emotions: {labels}.\n"
            "Based on psychological and visual features such as facial expression, body posture, "
            "background context, lighting, and colour tones, please analyse the emotional state "
            "represented.\n"
            f'Image Description: "{caption}"\n'
            "Think carefully and choose only one emotion word from the list above."
        )
    else:
        feedback = state.get("textual_feedback") if query_mode == "qtext" else state.get("visual_feedback") or {}
        suggestion = (feedback or {}).get("suggestion", prediction)
        fb_text    = (feedback or {}).get("justification", "")

        return (
            "Image Context for Current Task:\n"
            f'Image Caption: "{caption}"\n'
            "IMPORTANT GUIDANCE FOR REFINEMENT:\n"
            f'A previous reflection on the emotion "{prediction}" suggested the new emotion '
            f'might be "{suggestion}" based on the following feedback: "{fb_text}". '
            "Please carefully consider this feedback.\n"
            f"Your task is to choose the best emotion label for the image from the following list: {labels}. "
            "Output only the chosen emotion word."
        )

def run_decision_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    mode = state.get("query_mode", "qinit")
    print(f"--- [Agent: Decision | mode={mode}] Iteration {state.get('iteration', 0)} ---")

    prompt_text = _build_prompt(state)
    raw_output = run_llava_generation(prompt_text, state["image_path"], memory_manager, max_new_tokens=50)

    # Map raw output to a valid candidate
    prediction = next(
        (e for e in state["candidate_emotions"] if e.lower() in raw_output.lower()),
        state["candidate_emotions"][0]
    )

    print(f"    -> Prediction: '{prediction}' (raw: '{raw_output}')")
    return {"current_prediction": prediction}
