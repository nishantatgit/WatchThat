"""PyTorch datasets for unlabelled image training."""

import json
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


def load_dataset_manifest(
    manifest_path: str | Path,
) -> tuple[DatasetManifestItem, ...]:
    path = Path(manifest_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")

    items: list[DatasetManifestItem] = []

    with path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                data = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on manifest line {line_number}"
                ) from error

            if not isinstance(data, dict):
                raise TypeError(
                    f"Manifest line {line_number} must contain a JSON object"
                )

            try:
                item = DatasetManifestItem(
                    image_id=str(data["image_id"]),
                    storage_uri=str(data["storage_uri"]),
                    checksum=str(data["checksum"]),
                    content_type=str(data["content_type"]),
                    size_bytes=int(data["size_bytes"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid dataset item on manifest line {line_number}"
                ) from error

            items.append(item)

    if not items:
        raise ValueError("Dataset manifest does not contain any images")

    return tuple(items)
