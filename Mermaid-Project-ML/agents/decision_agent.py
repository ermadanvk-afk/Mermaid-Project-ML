import torch
from PIL import Image
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator


def _build_prompt(state: MERMAIDState) -> str:
    """
    Builds the correct text prompt for the Decision Agent based on query_mode.

    Three modes (Appendix C, MERMAID paper):
      - "qinit" : Base Emotion Classification Prompt — no feedback, first pass.
      - "qtext" : Augmentation prompt guided by textual_feedback.
      - "qvis"  : Augmentation prompt guided by visual_feedback.

    Returns:
        The full [INST] ... [/INST] prompt string for LLaVA-NeXT.
    """
    labels       = ", ".join(state["candidate_emotions"])
    caption      = state.get("caption", "")
    query_mode   = state.get("query_mode", "qinit")
    prediction   = state.get("current_prediction", "")

    if query_mode == "qinit":
        # ── Base Emotion Classification Prompt (paper, Appendix C) ──────────
        text = (
            f"This image must be classified into one of these human emotions: {labels}.\n"
            "Based on psychological and visual features such as facial expression, body posture, "
            "background context, lighting, and colour tones, please analyse the emotional state "
            "represented.\n"
            f'Image Description: "{caption}"\n'
            "Think carefully and choose only one emotion word from the list above."
        )

    else:
        # ── Emotion Classifier Prompt Augmentation (paper, Appendix C) ──────
        # Selects the correct feedback source depending on mode.
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

    return f"[INST] <image>\n{text} [/INST]"


def run_decision_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    """
    Core classifier node. Called three times per outer loop with different query modes:
      - Qinit (Block 1) : initial classification.
      - Qtext (Block 2) : re-classification guided by textual reflection feedback.
      - Qvis  (Block 3) : re-classification guided by visual reflection feedback.

    Input:
        state          – Current LangGraph state (query_mode determines prompt).
        memory_manager – Provides the 4-bit LLaVA-NeXT model and processor.

    Output:
        dict with key 'current_prediction' — one label from candidate_emotions.
    """
    mode = state.get("query_mode", "qinit")
    print(f"--- [Agent: Decision | mode={mode}] Iteration {state['iteration']} ---")

    # 1. Ensure MLLM is on GPU
    memory_manager.load_mllm()
    model     = memory_manager.mllm
    processor = memory_manager.processor

    # 2. Load image
    image = Image.open(state["image_path"]).convert("RGB")

    # 3. Build the mode-appropriate prompt
    prompt = _build_prompt(state)

    # 4. Run inference
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20)

    input_len  = inputs["input_ids"].shape[1]
    raw_output = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    # 5. Map raw output to a valid candidate (case-insensitive, fallback to first)
    prediction = next(
        (e for e in state["candidate_emotions"] if e.lower() in raw_output.lower()),
        state["candidate_emotions"][0],
    )

    print(f"    -> Predicted: '{prediction}'  (raw: '{raw_output}')")

    return {"current_prediction": prediction}

