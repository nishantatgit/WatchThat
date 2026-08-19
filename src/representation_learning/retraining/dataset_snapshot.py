"""Immutable and reproducible training dataset snapshots."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from representation_learning.domain import ImageStatus
from representation_learning.storage.metadata_store import MetadataStore


@dataclass(frozen=True, slots=True)
class DatasetManifestItem:
    image_id: str
    storage_uri: str
    checksum: str
    content_type: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "image_id": self.image_id,
            "storage_uri": self.storage_uri,
            "checksum": self.checksum,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    snapshot_id: str
    created_at: datetime
    items: tuple[DatasetManifestItem, ...]

    @property
    def image_count(self) -> int:
        return len(self.items)

    def to_jsonl(self) -> bytes:
        lines = [
            json.dumps(
                item.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in self.items
        ]

        return ("\n".join(lines) + "\n").encode("utf-8")


class DatasetSnapshotBuilder:
    def __init__(self, metadata_store: MetadataStore) -> None:
        self._metadata_store = metadata_store

    def build(self) -> DatasetSnapshot:
        records = [
            record
            for record in self._metadata_store.list_all()
            if record.status == ImageStatus.ACCEPTED
        ]

        if not records:
            raise ValueError("Cannot create a dataset snapshot without accepted images")

        items = tuple(
            DatasetManifestItem(
                image_id=record.image_id,
                storage_uri=record.storage_uri,
                checksum=record.checksum,
                content_type=record.content_type,
                size_bytes=record.size_bytes,
            )
            for record in sorted(
                records,
                key=lambda record: record.image_id,
            )
        )

        manifest_content = b"\n".join(
            json.dumps(
                item.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            for item in items
        )

        snapshot_id = sha256(manifest_content).hexdigest()[:16]

        return DatasetSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now(UTC),
            items=items,
        )


class DatasetSnapshotStore(Protocol):
    def save(self, snapshot: DatasetSnapshot) -> str: ...


class AzureBlobDatasetSnapshotStore:
    def __init__(
        self,
        *,
        account_url: str,
        container_name: str,
        credential: TokenCredential | None = None,
    ) -> None:
        self._blob_service = BlobServiceClient(
            account_url=account_url,
            credential=credential or DefaultAzureCredential(),
        )
        self._container_name = container_name

    def save(self, snapshot: DatasetSnapshot) -> str:
        blob_name = f"snapshots/{snapshot.snapshot_id}/manifest.jsonl"

        blob_client = self._blob_service.get_blob_client(
            container=self._container_name,
            blob=blob_name,
        )

        manifest = snapshot.to_jsonl()

        try:
            blob_client.upload_blob(
                manifest,
                overwrite=False,
                content_settings=ContentSettings(content_type="application/x-ndjson"),
                metadata={
                    "snapshot_id": snapshot.snapshot_id,
                    "image_count": str(snapshot.image_count),
                    "created_at": snapshot.created_at.isoformat(),
                },
            )
        except ResourceExistsError:
            existing_manifest = blob_client.download_blob().readall()

            if existing_manifest != manifest:
                raise RuntimeError(
                    "A different manifest already exists for "
                    f"snapshot {snapshot.snapshot_id}"
                )

        return blob_client.url
