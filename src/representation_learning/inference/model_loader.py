"""Version-aware encoder loading and lifecycle management."""

from dataclasses import dataclass
from pathlib import Path

import torch

from representation_learning.models.contrastive_model import (
    ContrastiveModel,
)
from representation_learning.models.encoder import ImageEncoder
from representation_learning.training.checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
)


@dataclass(frozen=True, slots=True)
class LoadedEncoder:
    encoder: ImageEncoder
    metadata: CheckpointMetadata
    architecture_version: str
    embedding_dimension: int
    device: torch.device


class EncoderLoader:
    _SUPPORTED_ARCHITECTURES = frozenset({"cnn-v1"})

    def load(
        self,
        *,
        checkpoint_path: str | Path,
        architecture_version: str,
        projection_dimension: int,
        device: torch.device | None = None,
    ) -> LoadedEncoder:
        path = Path(checkpoint_path)

        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

        if architecture_version not in self._SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported model architecture: {architecture_version}")

        selected_device = device or self._select_device()

        model = ContrastiveModel(
            projection_dimension=projection_dimension,
        )
        model.to(selected_device)

        checkpoint_manager = CheckpointManager(
            path.parent,
        )
        metadata = checkpoint_manager.load(
            checkpoint_path=path,
            model=model,
            device=selected_device,
        )

        encoder = model.encoder
        encoder.eval()
        encoder.requires_grad_(False)

        return LoadedEncoder(
            encoder=encoder,
            metadata=metadata,
            architecture_version=architecture_version,
            embedding_dimension=encoder.feature_dimension,
            device=selected_device,
        )

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
