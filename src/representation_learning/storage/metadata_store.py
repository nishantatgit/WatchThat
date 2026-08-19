"""Persistent image metadata and processing-status storage."""
# src/representation_learning/storage/metadata_store.py

from typing import Protocol
from datetime import datetime

from azure.core.credentials import TokenCredential
from azure.core.exceptions import (
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.data.tables import TableClient
from azure.identity import DefaultAzureCredential

from representation_learning.domain.entities import (
    ImageRecord,
    ImageSource,
    ImageStatus,
)

from representation_learning.domain.entities import ImageRecord


class MetadataStore(Protocol):
    def save(self, record: ImageRecord) -> None:
        ...

    def get(self, image_id: str) -> ImageRecord | None:
        ...

    def find_by_checksum(self, checksum: str) -> ImageRecord | None:
        ...

    def list_all(self) -> list[ImageRecord]:
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

class AzureTableMetadataStore:
    def __init__(
        self,
        *,
        endpoint: str,
        table_name: str,
        credential: TokenCredential | None = None,
    ) -> None:
        self._table = TableClient(
            endpoint=endpoint,
            table_name=table_name,
            credential=credential or DefaultAzureCredential(),
        )

    def save(self, record: ImageRecord) -> None:
        entity = {
            "PartitionKey": self._partition_key(record.checksum),
            "RowKey": record.checksum,
            "ImageId": record.image_id,
            "Source": record.source.value,
            "StorageUri": record.storage_uri,
            "ContentType": record.content_type,
            "Width": record.width,
            "Height": record.height,
            "SizeBytes": record.size_bytes,
            "Status": record.status.value,
            "CreatedAt": record.created_at,
        }

        try:
            # create_entity fails if this checksum already exists.
            self._table.create_entity(entity=entity)
        except ResourceExistsError as error:
            existing = self.find_by_checksum(record.checksum)
            existing_id = (
                existing.image_id if existing is not None else "unknown"
            )

            raise ValueError(
                "An image with the same checksum already exists: "
                f"{existing_id}"
            ) from error

    def get(self, image_id: str) -> ImageRecord | None:
        escaped_image_id = image_id.replace("'", "''")
        entities = self._table.query_entities(
            query_filter=f"ImageId eq '{escaped_image_id}'",
        )

        entity = next(iter(entities), None)

        if entity is None:
            return None

        return self._to_record(entity)

    def find_by_checksum(self, checksum: str) -> ImageRecord | None:
        try:
            entity = self._table.get_entity(
                partition_key=self._partition_key(checksum),
                row_key=checksum,
            )
        except ResourceNotFoundError:
            return None

        return self._to_record(entity)

    def list_all(self) -> list[ImageRecord]:
        return [
            self._to_record(entity)
            for entity in self._table.list_entities()
        ]

    # These methods also make this store satisfy ChecksumRegistry.

    def contains(self, checksum: str) -> bool:
        return self.find_by_checksum(checksum) is not None

    def get_image_id(self, checksum: str) -> str | None:
        record = self.find_by_checksum(checksum)
        return record.image_id if record is not None else None

    def add(self, checksum: str, image_id: str) -> None:
        record = self.find_by_checksum(checksum)

        if record is None:
            raise ValueError(
                "Metadata must be saved before registering its checksum"
            )

        if record.image_id != image_id:
            raise ValueError(
                f"Checksum belongs to a different image: {record.image_id}"
            )

    @staticmethod
    def _partition_key(checksum: str) -> str:
        if len(checksum) < 2:
            raise ValueError("Checksum must contain at least two characters")

        # 256 evenly distributed partitions for SHA-256 checksums.
        return checksum[:2]

    @staticmethod
    def _to_record(entity: dict) -> ImageRecord:
        created_at = entity["CreatedAt"]

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return ImageRecord(
            image_id=entity["ImageId"],
            source=ImageSource(entity["Source"]),
            storage_uri=entity["StorageUri"],
            checksum=entity["RowKey"],
            content_type=entity["ContentType"],
            width=int(entity["Width"]),
            height=int(entity["Height"]),
            size_bytes=int(entity["SizeBytes"]),
            status=ImageStatus(entity["Status"]),
            created_at=created_at,
        )