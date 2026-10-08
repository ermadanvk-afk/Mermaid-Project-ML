import json
import random
import torch
from collections import Counter
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.schemas import ReflectionOutput

# ── Perspectives from MERMAID paper, Appendix C ────────────────────────────
PERSPECTIVES = {
    "Default": (
        "You are an Emotion Assessment AI (Text-Only). "
        'Current prediction: "{current_emotion}". Caption: "{caption}". Labels: {labels}. '
        "Evaluate the prediction. If correct, confirm. Otherwise, suggest a better label and explain why."
    ),
    "Focus_expression": (
        "You are a facial expression analyst. "
        'Based on the facial clues in the caption: "{caption}", '
        'judge whether the emotion "{current_emotion}" is correct among: {labels}. '
        "Explain your reasoning clearly."
    ),
    "Focus_scene": (
        "You are a background context emotion specialist. "
        'Based on environmental and contextual elements from the caption: "{caption}", '
        'evaluate whether "{current_emotion}" is appropriate among: {labels}. '
        "Justify your reasoning."
    ),
    "Focus_color": (
        "As a colour psychologist, "
        'analyse if the description: "{caption}" supports "{current_emotion}" '
        "based on tone, colour, and mood. "
        "Choose the best label from: {labels}."
    ),
    "Focus_action": (
        "You are an emotional behaviour analyst. "
        'From the actions described in the caption: "{caption}", '
        'infer whether "{current_emotion}" is the most fitting label among: {labels}. '
        "Explain your reasoning."
    ),
    "Focus_social": (
        "You are a social context evaluator. "
        'Based on social interactions or cues described in the caption: "{caption}", '
        'assess if "{current_emotion}" is a reasonable label among: {labels}. '
        "Justify your answer."
    ),
}

# ── Base wrapper prompt from Appendix C (modified for JSON) ────────────────
_BASE_PROMPT_TEMPLATE = (
    "You are an Emotion Assessment AI acting in the role of a {perspective}, "
    "specialised in analysing emotion from textual descriptions.\n"
    'Current predicted emotion: "{current_emotion}"\n'
    'Caption of the query image: "{caption}"\n'
    "List of candidate emotion labels: {labels}\n"
    "Your Task: 1. Evaluate whether the predicted emotion \"{current_emotion}\" is the most "
    "appropriate given the caption. Rely solely on the textual description. "
    "2. If the prediction is appropriate, confirm it. "
    "3. If not, suggest a better emotion from the label list and provide a concise explanation.\n"
    "Output ONLY a valid JSON object matching this schema:\n"
    "{{\n"
    '  "status": "good_enough" or "needs_revision",\n'
    '  "suggestion": "<label>",\n'
    '  "justification": "<explanation>"\n'
    "}}\n"
    "Do not include any other text."
)


def _parse_json_output(raw: str, current_prediction: str, candidate_emotions: list) -> dict:
    """
    Extracts and validates JSON using the ReflectionOutput Pydantic schema.
    """
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            
        # The LLM frequently outputs "good\_enough" which is invalid JSON.
        raw = raw.replace('\\_', '_')
            
        # Validate against schema
        parsed = ReflectionOutput.model_validate_json(raw)
        
        # Ensure suggestion is valid
        suggestion = parsed.suggestion
        matched = next((e for e in candidate_emotions if e.lower() == suggestion.lower()), None)
        suggestion = matched if matched else current_prediction
        
        return {
            "status": parsed.status,
            "suggestion": suggestion,
            "justification": parsed.justification
        }
    except Exception as e:
        print(f"      [Warning] JSON parse failed: {e}. Raw: {raw}")
        return {
            "status": "needs_revision",
            "suggestion": current_prediction,
            "justification": "Failed to parse MLLM output."
        }


def _run_single_perspective(
    perspective_name: str,
    state: MERMAIDState,
    model,
    processor,
) -> dict:
    """Runs textual reflection for one perspective and returns parsed JSON."""

    labels     = ", ".join(state["candidate_emotions"])
    caption    = state.get("caption", "")
    prediction = state.get("current_prediction", "")

    # Fill the perspective role description
    role_text = PERSPECTIVES[perspective_name].format(
        current_emotion=prediction,
        caption=caption,
        labels=labels,
    )

    # Build the full prompt using the base wrapper from Appendix C
    instruction = _BASE_PROMPT_TEMPLATE.format(
        perspective=role_text,
        current_emotion=prediction,
        caption=caption,
        labels=labels,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Text-only inference
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    parsed = _parse_json_output(raw_output, prediction, state["candidate_emotions"])
    print(f"      [{perspective_name}] status={parsed['status']}, suggestion={parsed['suggestion']}")
    return parsed


def _aggregate_votes(votes: list[dict], current_prediction: str) -> dict:
    """
    Aggregates multi-perspective votes by majority:
      - status   : majority wins (tie → needs_revision to be safe)
      - suggestion: most-common suggestion among "needs_revision" voters
      - justification: concatenated, separated by ' | '
    """
    statuses    = [v["status"] for v in votes]
    status_vote = Counter(statuses).most_common(1)[0][0]

    statuses = [v["status"] for v in votes]
    
    # Among those voting needs_revision, find most common suggestion
    revisers = [v["suggestion"] for v in votes if v["status"] == "needs_revision"]
    if revisers:
        suggestion = Counter(revisers).most_common(1)[0][0]
    else:
        suggestion = current_prediction

    justifications = " | ".join(v["justification"] for v in votes)

    return {
        "status": status_vote,
        "suggestion": suggestion,
        "justification": justifications,
    }


def run_text_reflection(
    state: MERMAIDState,
    memory_manager: MemoryOrchestrator,
    num_perspectives: int = 3,
) -> dict:
    """
    Multi-perspective textual self-reflection agent (MERMAID paper, Section 4.3).

    Randomly samples `num_perspectives` perspectives from the defined set,
    runs each as a separate MLLM call (text-only), then aggregates via majority vote.

    Input:
        state            – LangGraph state with 'caption', 'current_prediction', 'candidate_emotions'.
        memory_manager   – Provides the 4-bit LLaVA-NeXT model and processor.
        num_perspectives – Number of perspectives to sample (default 3, per paper §5.3).

    Output:
        dict with key 'textual_feedback': {status, suggestion, justification}.
    """
    print(f"--- [Agent: Text Reflection | {num_perspectives} perspectives] ---")

    # 1. Ensure MLLM is on GPU
    memory_manager.load_mllm()
    model     = memory_manager.mllm
    processor = memory_manager.processor

    # 2. Sample perspectives — always include "Default", fill remainder randomly
    specialist_pool = [p for p in PERSPECTIVES if p != "Default"]
    sampled = ["Default"] + random.sample(specialist_pool, k=min(num_perspectives - 1, len(specialist_pool)))
    print(f"    -> Sampled perspectives: {sampled}")

    # 3. Run each perspective
    votes = []
    for perspective_name in sampled:
        vote = _run_single_perspective(perspective_name, state, model, processor)
        votes.append(vote)

    # 4. Aggregate by majority vote
    feedback = _aggregate_votes(votes, state.get("current_prediction", ""))
    print(f"    -> Aggregated: status={feedback['status']}, suggestion={feedback['suggestion']}")

    return {"textual_feedback": feedback}

