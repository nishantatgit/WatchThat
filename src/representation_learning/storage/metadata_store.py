"""Persistent image metadata and processing-status storage."""
# src/representation_learning/storage/metadata_store.py

from typing import Protocol

from representation_learning.domain.entities import ImageRecord


class MetadataStore(Protocol):
    def save(self, record: ImageRecord) -> None:
        ...

    def get(self, image_id: str) -> ImageRecord | None:
        ...

    def find_by_checksum(self, checksum: str) -> ImageRecord | None:
        ...


class InMemoryMetadataStore:
    def __init__(self) -> None:
        self._records_by_id: dict[str, ImageRecord] = {}
        self._image_ids_by_checksum: dict[str, str] = {}

    def save(self, record: ImageRecord) -> None:
        if record.image_id in self._records_by_id:
            raise ValueError(
                f"Image metadata already exists: {record.image_id}"
            )

        existing = self.find_by_checksum(record.checksum)

        if existing is not None:
            raise ValueError(
                "An image with the same checksum already exists: "
                f"{existing.image_id}"
            )

        self._records_by_id[record.image_id] = record
        self._image_ids_by_checksum[record.checksum] = record.image_id

    def get(self, image_id: str) -> ImageRecord | None:
        return self._records_by_id.get(image_id)

    def find_by_checksum(self, checksum: str) -> ImageRecord | None:
        image_id = self._image_ids_by_checksum.get(checksum)

        if image_id is None:
            return None

        return self._records_by_id[image_id]

    def list_all(self) -> list[ImageRecord]:
        return list(self._records_by_id.values())