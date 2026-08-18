"""Synchronous user-upload ingestion workflow."""
# src/representation_learning/ingestion/user_upload.py

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)

from representation_learning.domain import (
    DomainEvent,
    ImageAccepted,
    ImageQuarantined,
    ImageReceived,
    ImageRecord,
    ImageSource,
    ImageStatus,
)
from representation_learning.ingestion.deduplication import ImageDeduplicator
from representation_learning.ingestion.validation import ImageValidator
from representation_learning.storage.image_store import ImageStore, StorageArea
from representation_learning.storage.metadata_store import MetadataStore


_CONTENT_TYPE_TO_EXTENSION = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass(frozen=True, slots=True)
class UserUploadResult:
    image_id: str
    record: ImageRecord | None
    events: tuple[DomainEvent, ...]
    duplicate_of: str | None = None

    @property
    def accepted(self) -> bool:
        return self.record is not None

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None


class UserUploadIngestion:
    def __init__(
        self,
        *,
        validator: ImageValidator,
        deduplicator: ImageDeduplicator,
        image_store: ImageStore,
        metadata_store: MetadataStore,
    ) -> None:
        self._validator = validator
        self._deduplicator = deduplicator
        self._image_store = image_store
        self._metadata_store = metadata_store

    def ingest(
        self,
        *,
        image_bytes: bytes,
        original_filename: str,
    ) -> UserUploadResult:
        image_id = str(uuid4())

        received = ImageReceived(
            image_id=image_id,
            source=ImageSource.USER_UPLOAD,
            storage_uri=original_filename,
        )

        validation = self._validator.validate(image_bytes)

        if not validation.is_valid:
            extension = self._safe_extension(original_filename)

            quarantine_uri = self._image_store.save(
                image_id=image_id,
                content=image_bytes,
                area=StorageArea.QUARANTINE,
                extension=extension,
            )

            quarantined = ImageQuarantined(
                image_id=image_id,
                storage_uri=quarantine_uri,
                reason=validation.reason or "Unknown validation failure",
            )

            return UserUploadResult(
                image_id=image_id,
                record=None,
                events=(received, quarantined),
            )

        deduplication = self._deduplicator.check(image_bytes)

        if deduplication.is_duplicate:
            return UserUploadResult(
                image_id=image_id,
                record=None,
                events=(received,),
                duplicate_of=deduplication.existing_image_id,
            )

        content_type = validation.content_type

        if content_type is None:
            raise RuntimeError(
                "Successful validation did not produce a content type"
            )

        extension = _CONTENT_TYPE_TO_EXTENSION[content_type]

        storage_uri = self._image_store.save(
            image_id=image_id,
            content=image_bytes,
            area=StorageArea.ACCEPTED,
            extension=extension,
        )

        record = ImageRecord.create(
            image_id=image_id,
            source=ImageSource.USER_UPLOAD,
            storage_uri=storage_uri,
            checksum=deduplication.checksum,
            content_type=content_type,
            width=self._required_dimension(validation.width, "width"),
            height=self._required_dimension(validation.height, "height"),
            size_bytes=len(image_bytes),
            status=ImageStatus.ACCEPTED,
        )

        self._metadata_store.save(record)
        self._deduplicator.register(record.checksum, record.image_id)

        accepted = ImageAccepted(
            image_id=image_id,
            storage_uri=storage_uri,
            checksum=record.checksum,
        )

        return UserUploadResult(
            image_id=image_id,
            record=record,
            events=(received, accepted),
        )

    @staticmethod
    def _safe_extension(filename: str) -> str:
        extension = Path(filename).suffix.lower().lstrip(".")

        if extension.isalnum():
            return extension

        return "bin"

    @staticmethod
    def _required_dimension(value: int | None, name: str) -> int:
        if value is None:
            raise RuntimeError(
                f"Successful validation did not produce {name}"
            )

        return value

@dataclass(frozen=True, slots=True)
class UploadGrant:
    image_id: str
    blob_uri: str
    upload_url: str
    expires_at: datetime


class DirectUploadService:
    _ALLOWED_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})

    def __init__(
        self,
        *,
        account_url: str,
        credential: TokenCredential | None = None,
        raw_container: str = "raw-images",
        sas_lifetime: timedelta = timedelta(minutes=10),
    ) -> None:
        if sas_lifetime <= timedelta(0):
            raise ValueError("sas_lifetime must be positive")

        self._credential = credential or DefaultAzureCredential()
        self._blob_service = BlobServiceClient(
            account_url=account_url,
            credential=self._credential,
        )
        self._raw_container = raw_container
        self._sas_lifetime = sas_lifetime

    def create_upload_grant(self, original_filename: str) -> UploadGrant:
        extension = Path(original_filename).suffix.lower().lstrip(".")

        if extension not in self._ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension or '<none>'}"
            )

        image_id = str(uuid4())
        blob_name = f"{image_id}.{extension}"

        blob_client = self._blob_service.get_blob_client(
            container=self._raw_container,
            blob=blob_name,
        )

        now = datetime.now(timezone.utc)
        expires_at = now + self._sas_lifetime

        # Start slightly in the past to tolerate clock differences.
        delegation_key = self._blob_service.get_user_delegation_key(
            key_start_time=now - timedelta(minutes=5),
            key_expiry_time=expires_at,
        )

        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=self._raw_container,
            blob_name=blob_name,
            user_delegation_key=delegation_key,
            permission=BlobSasPermissions(
                create=True,
                write=True,
            ),
            start=now - timedelta(minutes=5),
            expiry=expires_at,
        )

        return UploadGrant(
            image_id=image_id,
            blob_uri=blob_client.url,
            upload_url=f"{blob_client.url}?{sas_token}",
            expires_at=expires_at,
        )