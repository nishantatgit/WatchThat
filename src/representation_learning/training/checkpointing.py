"""Model, optimizer and scheduler checkpoint management."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

from representation_learning.models.contrastive_model import (
    ContrastiveModel,
)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    epoch: int
    training_loss: float
    validation_loss: float
    dataset_snapshot_id: str


class CheckpointManager:
    def __init__(
        self,
        output_directory: str | Path,
    ) -> None:
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        *,
        model: ContrastiveModel,
        optimizer: Optimizer,
        metadata: CheckpointMetadata,
        filename: str = "best-checkpoint.pt",
    ) -> Path:
        checkpoint_path = self._output_directory / filename
        temporary_path = checkpoint_path.with_suffix(".tmp")

        checkpoint: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": asdict(metadata),
        }

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

        return checkpoint_path

    def load(
        self,
        *,
        checkpoint_path: str | Path,
        model: ContrastiveModel,
        optimizer: Optimizer | None = None,
        device: torch.device,
    ) -> CheckpointMetadata:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return CheckpointMetadata(
            **checkpoint["metadata"],
        )
