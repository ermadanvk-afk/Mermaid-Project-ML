import json
import re
import random
import torch
from PIL import Image
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
    "Focus_cultural": (
        "You are a cross-cultural emotion specialist. "
        'Accounting for diverse cultural norms and non-Western emotional expressions in the caption: "{caption}", '
        'evaluate whether "{current_emotion}" is appropriate among: {labels}. '
        "Justify your reasoning."
    ),
}

# ── Base wrapper prompt (B5: avoided prompt duplication, added valid JSON example) ──
_BASE_PROMPT_TEMPLATE = (
    "{perspective}\n\n"
    "Your Task:\n"
    "1. Evaluate whether the predicted emotion \"{current_emotion}\" is the most appropriate given the caption. Rely solely on the textual description.\n"
    "2. If the prediction is appropriate, confirm it with status \"good_enough\" and suggestion \"{current_emotion}\".\n"
    "3. If not, suggest a better emotion from the candidate list with status \"needs_revision\" and provide a concise explanation.\n\n"
    "Example valid JSON response:\n"
    "{{\n"
    '  "status": "needs_revision",\n'
    '  "suggestion": "Joy",\n'
    '  "justification": "The caption mentions laughter and cheering among celebrating friends, which indicates Joy rather than Sadness."\n'
    "}}\n\n"
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
    Safely extracts and validates JSON using the ReflectionOutput Pydantic schema.
    Fallback searches only in the substring after 'suggestion' with word boundaries
    and defaults to good_enough / current_prediction if parsing completely fails.
    """
    text = raw.replace('\\_', '_').strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Extract JSON object substring { ... }
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        # Validate against schema
        parsed = ReflectionOutput.model_validate_json(text)
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
        lower = raw.lower()

        if "good_enough" in lower or "good enough" in lower:
            status = "good_enough"
            suggestion = current_prediction
        elif "needs_revision" in lower or "needs revision" in lower:
            status = "needs_revision"
            suggestion = current_prediction
            # Search only in substring after "suggestion" using word boundaries \b
            if "suggestion" in lower:
                after_suggestion = raw[lower.find("suggestion"):]
                for emotion in candidate_emotions:
                    if re.search(rf"\b{re.escape(emotion)}\b", after_suggestion, re.IGNORECASE):
                        suggestion = emotion
                        break
        else:
            # If parsing completely fails, default to good_enough and current_prediction
            status = "good_enough"
            suggestion = current_prediction

        justification = raw[:120].strip() if raw else "Parsed via fallback."
        return {
            "status": status,
            "suggestion": suggestion,
            "justification": justification
        }


def _run_single_perspective(
    perspective_name: str,
    state: MERMAIDState,
    memory_manager: MemoryOrchestrator,
) -> dict:
    """Runs textual reflection for one perspective and returns parsed JSON."""

    labels     = ", ".join(state["candidate_emotions"])
    caption    = state.get("caption", "")
    prediction = state.get("current_prediction", "")

    # Fill the perspective role description (contains caption and labels once)
    role_text = PERSPECTIVES[perspective_name].format(
        current_emotion=prediction,
        caption=caption,
        labels=labels,
    )

    # Build prompt without duplicating caption and label list twice (B5 fix)
    instruction = _BASE_PROMPT_TEMPLATE.format(
        perspective=role_text,
        current_emotion=prediction,
    )

    # Load query image for LLaVA
    image_path = state.get("image_path")
    if image_path:
        raw_image = Image.open(image_path).convert("RGB")
        raw_image.thumbnail((512, 512), Image.LANCZOS)
    else:
        raw_image = Image.new("RGB", (336, 336), (255, 255, 255))

    # B5: max_new_tokens increased to 200, no_repeat_ngram_size removed
    raw_output = memory_manager.generate_mllm_response(
        image=raw_image,
        prompt_text=instruction,
        max_new_tokens=200,
        repetition_penalty=1.15,
        no_repeat_ngram_size=0,
    )

    parsed = _parse_json_output(raw_output, prediction, state["candidate_emotions"])
    print(f"      [{perspective_name}] status={parsed['status']}, suggestion={parsed['suggestion']}")
    return parsed


def _aggregate_votes(votes: list[dict], current_prediction: str) -> dict:
    """
    Aggregates multi-perspective votes by majority:
      - status   : majority wins (tie → needs_revision to be safe)
      - suggestion: most-common suggestion among "needs_revision" voters
      - justification: only concatenated from voters whose status matches winning status_vote (B4 fix)
    """
    statuses    = [v["status"] for v in votes]
    status_vote = Counter(statuses).most_common(1)[0][0]

    # Among those voting needs_revision, find most common suggestion
    revisers = [v["suggestion"] for v in votes if v["status"] == "needs_revision"]
    if revisers:
        suggestion = Counter(revisers).most_common(1)[0][0]
    else:
        suggestion = current_prediction

    # B4 fix: Only join justifications from voters whose status matches the winning status_vote
    matching_justifications = [v["justification"] for v in votes if v["status"] == status_vote]
    justifications = " | ".join(matching_justifications) if matching_justifications else ""

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
        memory_manager   – Provides the 4-bit LLaVA-NeXT 7B model and processor.
        num_perspectives – Number of perspectives to sample (default 3, per paper §5.3).

    Output:
        dict with key 'textual_feedback': {status, suggestion, justification}.
    """
    print(f"--- [Agent: Text Reflection | LLaVA 1.5 7B | {num_perspectives} perspectives] ---")

    # 1. Ensure LLaVA 1.5 7B is on GPU
    memory_manager.load_mllm()

    # 2. Sample perspectives (B9 fix: if cultural_mode, include Focus_cultural)
    cultural_mode = state.get("cultural_mode", False)
    specialist_pool = [p for p in PERSPECTIVES if p not in ("Default", "Focus_cultural")]
    if cultural_mode and "Focus_cultural" in PERSPECTIVES:
        random_specialist = random.sample(specialist_pool, k=1)[0]
        sampled = ["Default", "Focus_cultural", random_specialist]
    else:
        sampled = ["Default"] + random.sample(specialist_pool, k=min(num_perspectives - 1, len(specialist_pool)))

    print(f"    -> Sampled perspectives: {sampled}")

    # 3. Run each perspective
    votes = []
    for perspective_name in sampled:
        vote = _run_single_perspective(perspective_name, state, memory_manager)
        votes.append(vote)

    # 4. Aggregate by majority vote
    feedback = _aggregate_votes(votes, state.get("current_prediction", ""))
    print(f"    -> Aggregated: status={feedback['status']}, suggestion={feedback['suggestion']}")

    return {"textual_feedback": feedback}

