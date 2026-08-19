"""PyTorch datasets for unlabelled image training."""

from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import torch
from PIL import Image
from torch.utils.data import Dataset

from representation_learning.data.augmentations import (
    ContrastiveTransform,
)
from representation_learning.retraining.dataset_snapshot import (
    DatasetManifestItem,
)


class ImageReader(Protocol):
    def read(self, item: DatasetManifestItem) -> Image.Image: ...


class MountedBlobImageReader:
    def __init__(self, mount_directory: str | Path) -> None:
        self._mount_directory = Path(mount_directory).resolve()

    def read(self, item: DatasetManifestItem) -> Image.Image:
        url_path = unquote(urlparse(item.storage_uri).path)
        path_parts = url_path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid blob URI: {item.storage_uri}")

        # The first component is the container name.
        relative_blob_path = Path(*path_parts[1:])

        image_path = (self._mount_directory / relative_blob_path).resolve()

        if not image_path.is_relative_to(self._mount_directory):
            raise ValueError(f"Blob path escapes mount directory: {item.storage_uri}")

        if not image_path.is_file():
            raise FileNotFoundError(f"Mounted image does not exist: {image_path}")

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()


class ContrastiveImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        *,
        items: tuple[DatasetManifestItem, ...],
        image_reader: ImageReader,
        transform: ContrastiveTransform,
    ) -> None:
        if not items:
            raise ValueError("Dataset requires at least one image")

        self._items = items
        self._image_reader = image_reader
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        item = self._items[index]
        image = self._image_reader.read(item)
        first_view, second_view = self._transform(image)

        return first_view, second_view, item.image_id
