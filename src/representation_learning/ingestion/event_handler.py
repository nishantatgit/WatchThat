"""Idempotent handler for Blob Storage and Event Grid events."""

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

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


@dataclass(frozen=True, slots=True)
class BlobProcessingResult:
    image_id: str
    record: ImageRecord | None
    events: tuple[DomainEvent, ...]
    duplicate_of: str | None = None

    @property
    def accepted(self) -> bool:
        return self.record is not None

    @property
    def quarantined(self) -> bool:
        return any(isinstance(event, ImageQuarantined) for event in self.events)

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None


class BlobCreatedEventHandler:
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

    def handle(
        self,
        *,
        raw_blob_uri: str,
        source: ImageSource = ImageSource.USER_UPLOAD,
    ) -> BlobProcessingResult:
        image_id, extension = self._identity_from_uri(raw_blob_uri)

        received = ImageReceived(
            image_id=image_id,
            source=source,
            storage_uri=raw_blob_uri,
        )

        image_bytes = self._image_store.read(raw_blob_uri)
        validation = self._validator.validate(image_bytes)

        if not validation.is_valid:
            quarantine_uri = self._image_store.save(
                image_id=image_id,
                content=image_bytes,
                area=StorageArea.QUARANTINE,
                extension=extension,
            )

            self._image_store.delete(raw_blob_uri)

            quarantined = ImageQuarantined(
                image_id=image_id,
                storage_uri=quarantine_uri,
                reason=validation.reason or "Unknown validation failure",
            )

            return BlobProcessingResult(
                image_id=image_id,
                record=None,
                events=(received, quarantined),
            )

        deduplication = self._deduplicator.check(image_bytes)

        if deduplication.is_duplicate:
            self._image_store.delete(raw_blob_uri)

            return BlobProcessingResult(
                image_id=image_id,
                record=None,
                events=(received,),
                duplicate_of=deduplication.existing_image_id,
            )

        if (
            validation.content_type is None
            or validation.width is None
            or validation.height is None
        ):
            raise RuntimeError("Successful validation returned incomplete metadata")

        accepted_uri = self._image_store.save(
            image_id=image_id,
            content=image_bytes,
            area=StorageArea.ACCEPTED,
            extension=extension,
        )

        record = ImageRecord.create(
            image_id=image_id,
            source=source,
            storage_uri=accepted_uri,
            checksum=deduplication.checksum,
            content_type=validation.content_type,
            width=validation.width,
            height=validation.height,
            size_bytes=len(image_bytes),
            status=ImageStatus.ACCEPTED,
        )

        try:
            self._metadata_store.save(record)
            self._deduplicator.register(
                record.checksum,
                record.image_id,
            )
        except Exception:
            # Compensate for a metadata failure so we do not leave an
            # accepted blob without a corresponding metadata record.
            self._image_store.delete(accepted_uri)
            raise

        # Delete raw only after accepted blob and metadata both succeed.
        self._image_store.delete(raw_blob_uri)

        accepted = ImageAccepted(
            image_id=image_id,
            storage_uri=accepted_uri,
            checksum=record.checksum,
        )

        return BlobProcessingResult(
            image_id=image_id,
            record=record,
            events=(received, accepted),
        )

    @staticmethod
    def _identity_from_uri(storage_uri: str) -> tuple[str, str]:
        path = unquote(urlparse(storage_uri).path)
        filename = PurePosixPath(path).name
        file_path = PurePosixPath(filename)

        image_id = file_path.stem
        extension = file_path.suffix.lower().lstrip(".")

        if not image_id or not extension:
            raise ValueError(
                f"Blob URI does not contain a valid filename: {storage_uri}"
            )

        return image_id, extension
