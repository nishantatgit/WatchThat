"""NT-Xent loss implemented using PyTorch tensor operations."""

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def forward(
        self,
        first_projections: torch.Tensor,
        second_projections: torch.Tensor,
    ) -> torch.Tensor:
        if first_projections.shape != second_projections.shape:
            raise ValueError("Projection batches must have identical shapes")

        if first_projections.ndim != 2:
            raise ValueError("Projections must have shape [batch, dimension]")

        batch_size = first_projections.shape[0]

        if batch_size < 2:
            raise ValueError("Contrastive loss requires at least two images")

        first_normalized = F.normalize(
            first_projections,
            dim=1,
        )
        second_normalized = F.normalize(
            second_projections,
            dim=1,
        )

        projections = torch.cat(
            [first_normalized, second_normalized],
            dim=0,
        )

        logits = (projections @ projections.T) / self.temperature

        sample_count = 2 * batch_size

        self_mask = torch.eye(
            sample_count,
            device=logits.device,
            dtype=torch.bool,
        )

        logits = logits.masked_fill(
            self_mask,
            torch.finfo(logits.dtype).min,
        )

        indices = torch.arange(
            sample_count,
            device=logits.device,
        )

        positive_targets = (indices + batch_size) % sample_count

        return F.cross_entropy(
            logits,
            positive_targets,
        )
