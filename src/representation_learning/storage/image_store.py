"""Image-storage abstraction.

Implementations will support:
- Local filesystem for development
- Azure Blob Storage for production
"""
# src/representation_learning/storage/image_store.py

from enum import StrEnum
from pathlib import Path
from typing import Protocol

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
    ) -> str:
        """Store image bytes and return their storage URI."""
        ...
    def read(self, storage_uri: str) -> bytes:
        ...

    def delete(self, storage_uri: str) -> None:
        ...

class LocalImageStore:
    def __init__(self, root_directory: str | Path = "data") -> None:
        self._root_directory = Path(root_directory)

        for area in StorageArea:
            self._directory_for(area).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        image_id: str,
        content: bytes,
        area: StorageArea,
        extension: str,
    ) -> str:
        if not image_id.strip():
            raise ValueError("image_id cannot be empty")

        normalized_extension = extension.lower().lstrip(".")

        if not normalized_extension.isalnum():
            raise ValueError(f"Invalid file extension: {extension}")

        file_path = self._directory_for(area) / (
            f"{image_id}.{normalized_extension}"
        )

        # "xb" prevents accidentally overwriting an existing image.
        with file_path.open("xb") as image_file:
            image_file.write(content)

        return file_path.as_posix()

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

        self._credential = credential or DefaultAzureCredential()
        self._blob_service = BlobServiceClient(
            account_url=account_url,
            credential=self._credential,
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

    @staticmethod
    def _content_type(extension: str) -> str:
        content_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }

        return content_types.get(extension, "application/octet-stream")