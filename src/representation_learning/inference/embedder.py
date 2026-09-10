"""Batch and single-image embedding generation."""

"""
PIL images
   ↓ InferenceTransform
Individual [3, 224, 224] tensors
   ↓ torch.stack
Batch [N, 3, 224, 224]
   ↓ encoder
Features [N, 512]
   ↓ L2 normalization
Unit-length embeddings [N, 512]
"""

from collections.abc import Sequence

import torch
from PIL import Image
from torch.nn import functional as F

from representation_learning.data.augmentations import (
    InferenceTransform,
)
from representation_learning.models.encoder import ImageEncoder


class ImageEmbedder:
    def __init__(
        self,
        *,
        encoder: ImageEncoder,
        transform: InferenceTransform,
        device: torch.device,
    ) -> None:
        self._encoder = encoder
        self._transform = transform
        self._device = device

        self._encoder.to(device)
        self._encoder.eval()

    def embed(self, image: Image.Image) -> torch.Tensor:
        embeddings = self.embed_batch((image,))
        return embeddings[0]

    def embed_batch(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required")

        tensors = [self._transform(image) for image in images]

        batch = torch.stack(tensors).to(self._device)

        with torch.inference_mode():
            features = self._encoder(batch)
            embeddings = F.normalize(
                features,
                p=2,
                dim=1,
            )

        return embeddings.cpu()
