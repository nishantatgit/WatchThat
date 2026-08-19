"""Projection head used only during contrastive training."""

import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int = 512,
        hidden_dimension: int = 512,
        output_dimension: int = 128,
    ) -> None:
        super().__init__()

        dimensions = (
            input_dimension,
            hidden_dimension,
            output_dimension,
        )

        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("All dimensions must be positive")

        self.output_dimension = output_dimension

        self.layers = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dimension,
                output_dimension,
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Projection head expects a [batch, features] tensor")

        return self.layers(features)
