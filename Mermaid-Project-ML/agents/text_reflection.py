import json
import re
import random
import torch
from collections import Counter
from core.state import MERMAIDState
from core.memory_manager import MemoryOrchestrator
from core.schemas import ReflectionOutput
from agents.caption_agent import run_llava_generation

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

# ── Base wrapper prompt from Appendix C (JSON Output) ───────────────────────
_BASE_PROMPT_TEMPLATE = (
    "You are an Emotion Assessment AI acting in the role of a {perspective}, "
    "specialised in analysing emotion from textual descriptions.\n"
    'Current predicted emotion: "{current_emotion}"\n'
    'Caption of the query image: "{caption}"\n'
    "List of candidate emotion labels: {labels}\n"
    'Your Task: 1. Evaluate whether the predicted emotion "{current_emotion}" is the most '
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
    Robust JSON parser with regex fallback.
    Crucial: If the reflection suggests the same emotion as current_prediction,
    or confirms it, status is normalized to 'good_enough' so it exits immediately.
    """
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
    cleaned = cleaned.replace('\\_', '_')
    
    result = None
    # Attempt 1: Pydantic validation
    try:
        parsed = ReflectionOutput.model_validate_json(cleaned)
        matched = next((e for e in candidate_emotions if e.lower() == parsed.suggestion.lower()), None)
        result = {
            "status": parsed.status,
            "suggestion": matched if matched else current_prediction,
            "justification": parsed.justification
        }
    except Exception:
        pass

    # Attempt 2: Regex extraction
    if not result:
        try:
            status_m = re.search(r'"status"\s*:\s*"(good_enough|needs_revision)"', cleaned, re.IGNORECASE)
            sugg_m = re.search(r'"suggestion"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
            just_m = re.search(r'"justification"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
            if status_m:
                status = status_m.group(1).lower()
                raw_sugg = sugg_m.group(1) if sugg_m else current_prediction
                matched = next((e for e in candidate_emotions if e.lower() == raw_sugg.lower()), None)
                result = {
                    "status": status,
                    "suggestion": matched if matched else current_prediction,
                    "justification": just_m.group(1) if just_m else "Parsed via regex fallback."
                }
        except Exception:
            pass

    # Attempt 3: Keyword fallback
    if not result:
        lower = cleaned.lower()
        status = "good_enough" if "good_enough" in lower else "needs_revision"
        matched_e = next((e for e in candidate_emotions if e.lower() in lower), None)
        result = {
            "status": status,
            "suggestion": matched_e if matched_e else current_prediction,
            "justification": "Fallback parse from model output."
        }

    # CRITICAL LOGIC FIX:
    # If the reflection suggestion matches the current prediction,
    # the agent agrees! Mark as 'good_enough' to avoid needless diffusion loops.
    if result["suggestion"].lower() == current_prediction.lower():
        result["status"] = "good_enough"

    return result

def _run_single_perspective(
    perspective_name: str,
    state: MERMAIDState,
    memory_manager: MemoryOrchestrator,
) -> dict:
    labels     = ", ".join(state["candidate_emotions"])
    caption    = state.get("caption", "")
    prediction = state.get("current_prediction", "")

    role_text = PERSPECTIVES[perspective_name].format(
        current_emotion=prediction,
        caption=caption,
        labels=labels,
    )

    instruction = _BASE_PROMPT_TEMPLATE.format(
        perspective=role_text,
        current_emotion=prediction,
        caption=caption,
        labels=labels,
    )

    raw_output = run_llava_generation(instruction, None, memory_manager, max_new_tokens=300)
    parsed = _parse_json_output(raw_output, prediction, state["candidate_emotions"])
    print(f"      [{perspective_name}] status={parsed['status']}, suggestion={parsed['suggestion']}")
    return parsed

def _aggregate_votes(votes: list[dict], current_prediction: str) -> dict:
    statuses    = [v["status"] for v in votes]
    status_vote = Counter(statuses).most_common(1)[0][0]

    revisers = [v["suggestion"] for v in votes if v["status"] == "needs_revision"]
    suggestion = Counter(revisers).most_common(1)[0][0] if revisers else current_prediction
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
    print(f"--- [Agent: Text Reflection | {num_perspectives} perspectives] ---")

    specialist_pool = [p for p in PERSPECTIVES if p != "Default"]
    sampled = ["Default"] + random.sample(specialist_pool, k=min(num_perspectives - 1, len(specialist_pool)))
    print(f"    -> Sampled perspectives: {sampled}")

    votes = []
    for perspective_name in sampled:
        vote = _run_single_perspective(perspective_name, state, memory_manager)
        votes.append(vote)

    feedback = _aggregate_votes(votes, state.get("current_prediction", ""))
    print(f"    -> Aggregated: status={feedback['status']}, suggestion={feedback['suggestion']}")

    return {"textual_feedback": feedback}
