from __future__ import annotations

from typing import TypedDict, List, Dict, Optional, Literal

FeedbackStatus = Literal["good_enough", "needs_revision"]
QueryMode = Literal["qinit", "qtext", "qvis"]

class FeedbackDict(TypedDict):
    """
    Structured feedback returned by the reflection agents.

    Fields:
        status: whether the current decision is acceptable
        suggestion: optional improvement or alternative emotion candidate
        justification: explanation for why the decision passed or failed
    """
    status: FeedbackStatus
    suggestion: str
    justification: str


class MERMAIDState(TypedDict):
    """
    Main state dictionary shared across the LangGraph workflow.

    This object represents a single inference request and is updated by the
    captioning, decision, textual reflection, augmentation, and visual
    reflection nodes.
    """

    # Inputs
    image_path: str
    candidate_emotions: List[str]

    # Pre-computation
    caption: str

    # Iteration tracking
    iteration: int
    max_iterations: int

    # Current state
    current_prediction: Optional[str]

    # Tracks which query context the Decision Agent should use on next call:
    #   "qinit" → initial classification prompt (no feedback)
    #   "qtext" → guided by textual reflection feedback
    #   "qvis"  → guided by visual reflection feedback
    query_mode: QueryMode

    # Feedback data
    textual_feedback: Optional[FeedbackDict]
    visual_feedback: Optional[FeedbackDict]

    # Generated assets
    reference_images: Dict[str, str]


def create_initial_state(
    image_path: str,
    candidate_emotions: List[str],
    max_iterations: int = 3,
) -> MERMAIDState:
    """
    Initialize the state for a new image inference request.

    This function creates the starting state before the captioning node runs.
    It ensures the iteration counter begins at zero and the feedback fields
    are empty until the agents produce results.
    """
    if not image_path:
        raise ValueError("image_path cannot be empty.")

    if not candidate_emotions:
        raise ValueError("candidate_emotions must contain at least one emotion.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")

    return {
        "image_path": image_path,
        "candidate_emotions": list(candidate_emotions),
        "caption": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "current_prediction": None,
        "query_mode": "qinit",
        "textual_feedback": None,
        "visual_feedback": None,
        "reference_images": {},
    }


def set_prediction(state: MERMAIDState, emotion: str) -> None:
    """
    Set the current prediction for the state.

    This function enforces the architectural rule that the prediction must
    be one of the allowed candidate emotions.
    """
    if emotion not in state["candidate_emotions"]:
        raise ValueError(
            f"Emotion '{emotion}' is not in candidate_emotions: "
            f"{state['candidate_emotions']}"
        )

    state["current_prediction"] = emotion


def set_textual_feedback(
    state: MERMAIDState,
    status: FeedbackStatus,
    suggestion: str,
    justification: str,
) -> None:
    """
    Store the result of the textual self-reflection pass.

    This is called after the caption and current prediction are compared.
    The feedback determines whether the workflow should continue to
    augmentation or end early.
    """
    state["textual_feedback"] = {
        "status": status,
        "suggestion": suggestion,
        "justification": justification,
    }


def set_visual_feedback(
    state: MERMAIDState,
    status: FeedbackStatus,
    suggestion: str,
    justification: str,
) -> None:
    """
    Store the result of the visual self-reflection pass.

    This reflects the comparison between the original image and the generated
    reference images. It becomes the criterion for accepting the prediction
    or repeating the decision loop.
    """
    state["visual_feedback"] = {
        "status": status,
        "suggestion": suggestion,
        "justification": justification,
    }


def set_caption(state: MERMAIDState, caption: str) -> None:
    """
    Store the generated caption for the input image.

    The caption acts as the semantic foundation for the decision node and the
    textual reflection step.
    """
    if not isinstance(caption, str):
        raise TypeError("caption must be a string.")

    state["caption"] = caption.strip()


def add_reference_image(state: MERMAIDState, emotion: str, image_path: str) -> None:
    """
    Save a generated reference image for a specific emotion.

    This keeps track of all candidate-emotion visual references created during
    the augmentation step.
    """
    if emotion not in state["candidate_emotions"]:
        raise ValueError(f"Unknown emotion '{emotion}' for this state.")

    if not image_path:
        raise ValueError("image_path cannot be empty for a generated reference image.")

    state["reference_images"][emotion] = image_path


def increment_iteration(state: MERMAIDState) -> None:
    """
    Move to the next refinement loop iteration.

    This is used when the current prediction is not yet accepted and the system
    must revisit the decision node with updated feedback.
    """
    state["iteration"] += 1


def clear_feedback(state: MERMAIDState) -> None:
    """
    Reset feedback fields before the next round of reflective evaluation.

    This is useful when a new decision loop begins and the previous feedback
    should be discarded.
    """
    state["textual_feedback"] = None
    state["visual_feedback"] = None


def is_terminal_state(state: MERMAIDState) -> bool:
    """
    Return True when the workflow should stop.

    The system ends early if either reflection layer marks the current
    decision as good enough, or if the iteration count reaches the maximum.
    """
    if state["textual_feedback"] is not None:
        if state["textual_feedback"]["status"] == "good_enough":
            return True

    if state["visual_feedback"] is not None:
        if state["visual_feedback"]["status"] == "good_enough":
            return True

    if state["iteration"] >= state["max_iterations"]:
        return True

    return False


def validate_state(state: MERMAIDState) -> None:
    """
    Validate that the state contains all required keys and coherent values.

    This acts as a functional guard before the graph executes a node,
    preventing invalid transitions and data corruption.
    """
    required_fields = {
        "image_path": str,
        "candidate_emotions": list,
        "caption": str,
        "iteration": int,
        "max_iterations": int,
        "current_prediction": (type(None), str),
        "textual_feedback": (type(None), dict),
        "visual_feedback": (type(None), dict),
        "reference_images": dict,
    }

    for key, expected_type in required_fields.items():
        if key not in state:
            raise KeyError(f"Missing required state field: '{key}'")

        value = state[key]
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"State field '{key}' must be one of {expected_type}, got {type(value)}"
                )
        elif not isinstance(value, expected_type):
            raise TypeError(
                f"State field '{key}' must be of type {expected_type}, got {type(value)}"
            )

    if not state["candidate_emotions"]:
        raise ValueError("candidate_emotions cannot be empty.")

    if state["iteration"] < 0:
        raise ValueError("iteration cannot be negative.")

    if state["max_iterations"] <= 0:
        raise ValueError("max_iterations must be positive.")

    if state["current_prediction"] is not None:
        if state["current_prediction"] not in state["candidate_emotions"]:
            raise ValueError(
                f"current_prediction '{state['current_prediction']}' is not in candidate_emotions."
            )

    if state["textual_feedback"] is not None:
        required_feedback_keys = {"status", "suggestion", "justification"}
        if set(state["textual_feedback"].keys()) != required_feedback_keys:
            raise ValueError("textual_feedback must contain status, suggestion, and justification.")

    if state["visual_feedback"] is not None:
        required_feedback_keys = {"status", "suggestion", "justification"}
        if set(state["visual_feedback"].keys()) != required_feedback_keys:
            raise ValueError("visual_feedback must contain status, suggestion, and justification.")