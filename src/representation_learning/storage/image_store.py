"""Image-storage abstraction.

Implementations will support:
- Local filesystem for development
- Azure Blob Storage for production
"""
# src/representation_learning/storage/image_store.py

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, unquote

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobClient,
    BlobServiceClient,
    ContentSettings,
)


class StorageArea(StrEnum):
    RAW = "raw"
    ACCEPTED = "accepted"
    QUARANTINE = "quarantine"


class ImageStore(Protocol):
    def save(
        self,
        *,
        image_id: str,
        content: bytes,
        area: StorageArea,
        extension: str,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Store image bytes and return their storage URI."""
        ...

    def read(self, storage_uri: str) -> bytes: ...

    def delete(self, storage_uri: str) -> None: ...

    def get_metadata(
        self,
        storage_uri: str,
    ) -> dict[str, str]: ...


class LocalImageStore:
    def __init__(self, root_directory: str | Path = "data") -> None:
        self._root_directory = Path(root_directory)
        self._metadata_by_uri: dict[str, dict[str, str]] = {}
        for area in StorageArea:
            self._directory_for(area).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        image_id: str,
        content: bytes,
        area: StorageArea,
        extension: str,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        if not image_id.strip():
            raise ValueError("image_id cannot be empty")

        normalized_extension = extension.lower().lstrip(".")

        if not normalized_extension.isalnum():
            raise ValueError(f"Invalid file extension: {extension}")

        file_path = self._directory_for(area) / (f"{image_id}.{normalized_extension}")

        # "xb" prevents accidentally overwriting an existing image.
        with file_path.open("xb") as image_file:
            image_file.write(content)

        storage_uri = file_path.as_posix()
        self._metadata_by_uri[storage_uri] = dict(metadata or {})

        return storage_uri

    def read(self, storage_uri: str) -> bytes:
        file_path = Path(storage_uri)

        if not file_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {storage_uri}")

        return file_path.read_bytes()

    def _directory_for(self, area: StorageArea) -> Path:
        return self._root_directory / area.value

    def delete(self, storage_uri: str) -> None:
        file_path = Path(storage_uri)

        if file_path.exists():
            file_path.unlink()

        self._metadata_by_uri.pop(storage_uri, None)

    def get_metadata(
        self,
        storage_uri: str,
    ) -> dict[str, str]:
        return dict(self._metadata_by_uri.get(storage_uri, {}))


class AzureBlobImageStore:
    def __init__(
        self,
        *,
        account_url: str,
        credential: TokenCredential | None = None,
        container_names: dict[StorageArea, str] | None = None,
    ) -> None:
        if not account_url.strip():
            raise ValueError("account_url cannot be empty")

        block_size = 4 * 1024 * 1024

        self._credential = credential or DefaultAzureCredential()
        self._blob_service = BlobServiceClient(
            account_url=account_url,
            credential=self._credential,
            max_block_size=block_size,
            connection_timeout=120,
            read_timeout=120,
        )
        self._container_names = container_names or {
            StorageArea.RAW: "raw-images",
            StorageArea.ACCEPTED: "accepted-images",
            StorageArea.QUARANTINE: "quarantined-images",
        }

    def save(
        self,
        *,
        image_id: str,
        content: bytes,
        area: StorageArea,
        metadata: Mapping[str, str] | None = None,
        extension: str,
    ) -> str:
        if not image_id.strip():
            raise ValueError("image_id cannot be empty")

        normalized_extension = extension.lower().lstrip(".")

        if not normalized_extension.isalnum():
            raise ValueError(f"Invalid file extension: {extension}")

        blob_name = f"{image_id}.{normalized_extension}"
        container_name = self._container_names[area]

        blob_client = self._blob_service.get_blob_client(
            container=container_name,
            blob=blob_name,
        )

        blob_client.upload_blob(
            content,
            overwrite=False,
            content_settings=ContentSettings(
                content_type=self._content_type(normalized_extension)
            ),
            metadata=self._encode_metadata(metadata or {}),
        )

        return blob_client.url

    def read(self, storage_uri: str) -> bytes:
        blob_client = BlobClient.from_blob_url(
            blob_url=storage_uri,
            credential=self._credential,
        )

        return blob_client.download_blob().readall()

    def delete(self, storage_uri: str) -> None:
        blob_client = BlobClient.from_blob_url(
            blob_url=storage_uri,
            credential=self._credential,
        )

        blob_client.delete_blob(
            delete_snapshots="include",
        )

    def get_metadata(
        self,
        storage_uri: str,
    ) -> dict[str, str]:
        blob_client = BlobClient.from_blob_url(
            blob_url=storage_uri,
            credential=self._credential,
        )

        properties = blob_client.get_blob_properties()

        return {
            key: unquote(value) for key, value in (properties.metadata or {}).items()
        }

    @staticmethod
    def _encode_metadata(
        metadata: Mapping[str, str],
    ) -> dict[str, str]:
        return {
            key.casefold(): quote(value, safe="") for key, value in metadata.items()
        }

    @staticmethod
    def _content_type(extension: str) -> str:
        content_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }

        return content_types.get(extension, "application/octet-stream")
