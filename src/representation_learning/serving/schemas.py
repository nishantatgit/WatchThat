"""API request and response schemas."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimilarImage:
    image_id: str
    storage_uri: str
    similarity_score: float
    model_version: str
