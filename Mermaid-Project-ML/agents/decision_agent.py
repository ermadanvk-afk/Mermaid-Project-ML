import re
import json
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
            "Think carefully and choose only one emotion word from the list above.\n"
            'Respond with ONLY a valid JSON object: {"emotion": "<chosen_label>"}'
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
            f"{labels}.\n"
            'Respond with ONLY a valid JSON object: {"emotion": "<chosen_label>"}'
        )

    return text


def _parse_decision_output(
    raw_output: str,
    candidate_emotions: list[str],
    current_prediction: str = None,
    iteration: int = 0,
) -> str:
    """
    Parses decision agent output (B2 fix):
    a) First tries parsing JSON {"emotion": "<chosen_label>"}.
    b) Fallback: finds ALL occurrences of candidate emotions in generated response using
       word-boundary regex (\\b<label>\\b) and returns the LAST occurring candidate emotion
       in the text (never the first in candidate_emotions list order!).
    c) If zero labels match in a revision step (iteration > 0), retains current_prediction
       instead of blindly defaulting to candidate_emotions[0].
    """
    # 1. First try parsing JSON
    text = raw_output.replace('\\_', '_').strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and "emotion" in parsed:
                chosen = str(parsed["emotion"]).strip()
                for e in candidate_emotions:
                    if e.lower() == chosen.lower():
                        return e
        except Exception:
            pass

    # 2. Word-boundary regex search across generated response
    matches = []
    for emotion in candidate_emotions:
        pattern = rf"\b{re.escape(emotion)}\b"
        for m in re.finditer(pattern, raw_output, re.IGNORECASE):
            matches.append((m.start(), emotion))

    if matches:
        # Sort by character offset and pick the LAST occurring candidate emotion
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]

    # 3. Fallback for zero matches: if iteration > 0, retain current_prediction
    if iteration > 0 and current_prediction:
        return current_prediction
    return candidate_emotions[0]


def run_decision_agent(state: MERMAIDState, memory_manager: MemoryOrchestrator) -> dict:
    mode = state.get("query_mode", "qinit")
    iteration = state.get("iteration", 0)
    current_pred = state.get("current_prediction")
    candidate_emotions = state["candidate_emotions"]
    print(f"--- [Agent: Decision | mode={mode}] Iteration {iteration} (LLaVA 1.5 7B) ---")

    # 1. Load image and cap resolution (B7 fix: max dim 512x512, max_pixels <= 262144)
    image_path = state["image_path"]
    raw_image = Image.open(image_path).convert("RGB")
    raw_image.thumbnail((512, 512), Image.LANCZOS)
    prompt_text = _build_prompt(state)

    # 2. Generate response via LLaVA 1.5 7B chat-template pipeline
    raw_output = memory_manager.generate_mllm_response(
        image=raw_image,
        prompt_text=prompt_text,
        max_new_tokens=60,
        repetition_penalty=1.15,
        no_repeat_ngram_size=0,
    )

    # 3. Map raw output using B2 two-layer parser
    prediction = _parse_decision_output(
        raw_output=raw_output,
        candidate_emotions=candidate_emotions,
        current_prediction=current_pred,
        iteration=iteration,
    )

    print(f"    -> Predicted: '{prediction}'  (raw: '{raw_output.strip()}')")
    return {"current_prediction": prediction}

