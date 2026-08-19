"""Composition of the image encoder and projection head."""

from dataclasses import dataclass

import torch
from torch import nn

from representation_learning.models.encoder import ImageEncoder
from representation_learning.models.projection_head import ProjectionHead


@dataclass(frozen=True, slots=True)
class ContrastiveModelOutput:
    first_features: torch.Tensor
    second_features: torch.Tensor
    first_projections: torch.Tensor
    second_projections: torch.Tensor


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        *,
        projection_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = ImageEncoder()
        self.projection_head = ProjectionHead(
            input_dimension=self.encoder.feature_dimension,
            hidden_dimension=self.encoder.feature_dimension,
            output_dimension=projection_dimension,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(
        self,
        first_views: torch.Tensor,
        second_views: torch.Tensor,
    ) -> ContrastiveModelOutput:
        if first_views.shape != second_views.shape:
            raise ValueError("The two view batches must have identical shapes")

        first_features = self.encoder(first_views)
        second_features = self.encoder(second_views)

        first_projections = self.projection_head(first_features)
        second_projections = self.projection_head(second_features)

        return ContrastiveModelOutput(
            first_features=first_features,
            second_features=second_features,
            first_projections=first_projections,
            second_projections=second_projections,
        )
