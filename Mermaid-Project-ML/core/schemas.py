from typing import Literal, List

from pydantic import BaseModel, Field, field_validator


FeedbackStatus = Literal["good_enough", "needs_revision"]


class CaptionOutput(BaseModel):
    caption: str = Field(
        min_length=1,
        description="A rich textual description of the image contents, mood, and context.",
    )


class DecisionOutput(BaseModel):
    prediction: str = Field(
        min_length=1,
        description="A single emotion label chosen from the candidate_emotions list.",
    )

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prediction cannot be empty")
        return value


class ReflectionOutput(BaseModel):
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
    emotion: str = Field(
        min_length=1,
        description="Emotion label associated with the generated image.",
    )
    image_path: str = Field(
        min_length=1,
        description="Filesystem path to the generated reference image.",
    )


class AugmentationBatchOutput(BaseModel):
    reference_images: List[ReferenceImageOutput] = Field(
        description="List of generated reference image artifacts, one per candidate emotion."
    )

    @field_validator("reference_images")
    @classmethod
    def validate_reference_images(cls, value: List[ReferenceImageOutput]) -> List[ReferenceImageOutput]:
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

