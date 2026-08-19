"""Stochastic training and deterministic inference transformations."""

from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms


class ContrastiveTransform:
    def __init__(self, image_size: int = 224) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )

        self._transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.2, 1.0),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [color_jitter],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(
                    kernel_size=23,
                    sigma=(0.1, 2.0),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_image = image.convert("RGB")

        first_view = self._transform(rgb_image)
        second_view = self._transform(rgb_image)

        return first_view, second_view
