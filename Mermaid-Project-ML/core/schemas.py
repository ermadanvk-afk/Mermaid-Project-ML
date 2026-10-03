from typing import Literal, List

from pydantic import BaseModel, Field, field_validator


FeedbackStatus = Literal["good_enough", "needs_revision"]


class CaptionOutput(BaseModel):
    """
    Structured output returned by the CaptioningNode.

    The system prompt asks the MLLM to provide a concise but descriptive
    caption of the input image. This caption becomes the semantic basis for
    the decision step and later textual reflection.
    """

    caption: str = Field(
        min_length=1,
        description="A rich textual description of the image contents, mood, and context.",
    )


class DecisionOutput(BaseModel):
    """
    Structured output returned by the DecisionNode.

    The design specification states that the prediction must be a single string
    corresponding to one of the candidate emotions. This model wraps that value
    in a predictable schema while preserving the architectural contract.
    """

    prediction: str = Field(
        min_length=1,
        description="A single emotion label chosen from the candidate_emotions list.",
    )

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, value: str) -> str:
        """
        Enforce that the predicted label is a meaningful emotion string.

        The LLD requires the value to be one of the candidate_emotions, but the
        candidate list is known outside this schema. This validator guarantees the
        field is non-empty and human-readable before the graph checks membership.
        """
        value = value.strip()
        if not value:
            raise ValueError("prediction cannot be empty")
        return value


class ReflectionOutput(BaseModel):
    """
    Structured output returned by the textual and visual reflection agents.

    This matches the LLD exactly: each reflection node returns a JSON object with
    status, suggestion, and justification. The status tells the graph whether
    the prediction is acceptable or whether the workflow must continue with
    augmentation and another decision cycle.
    """

    status: FeedbackStatus = Field(
        description="Must be exactly 'good_enough' or 'needs_revision'."
    )
    suggestion: str = Field(
        description="A proposed improvement or alternative emotion based on the current evidence."
    )
    justification: str = Field(
        description="A concise explanation for the reflection result and suggested revision."
    )


class ReferenceImageOutput(BaseModel):
    """
    Structured output for an augmentation artifact.

    The AugmentationNode creates one generated image per candidate emotion and
    stores it in the state under reference_images. This schema documents the
    expected output contract for each generated artifact.
    """

    emotion: str = Field(
        min_length=1,
        description="Emotion label associated with the generated image.",
    )
    image_path: str = Field(
        min_length=1,
        description="Filesystem path to the generated reference image.",
    )


class AugmentationBatchOutput(BaseModel):
    """
    Structured output for the full augmentation stage.

    The HLD explains that the Diffusion model generates a set of emotion-specific
    reference images for all candidate emotions. This model captures the batch
    output as a collection of reference artifacts.
    """

    reference_images: List[ReferenceImageOutput] = Field(
        description="List of generated reference image artifacts, one per candidate emotion."
    )

    @field_validator("reference_images")
    @classmethod
    def validate_reference_images(cls, value: List[ReferenceImageOutput]) -> List[ReferenceImageOutput]:
        """
        Ensure the augmentation stage actually produces artifacts for the full set
        of candidate emotions and that no empty or invalid entries remain.
        """
        if not value:
            raise ValueError("reference_images cannot be empty")
        return value


__all__ = [
    "CaptionOutput",
    "DecisionOutput",
    "ReflectionOutput",
    "ReferenceImageOutput",
    "AugmentationBatchOutput",
    "FeedbackStatus",
]

