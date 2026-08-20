"""Core domain entities.

Planned entities:
- ImageRecord
- ImageSource
- ImageFeatureRecord
- EmbeddingRecord
- DatasetSnapshot
- ModelVersion
- ProcessingStatus
"""

# src/representation_learning/domain/entities.py

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ImageSource(StrEnum):
    USER_UPLOAD = "user_upload"
    WEB_SCRAPER = "web_scraper"


class ImageStatus(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    source: ImageSource
    storage_uri: str
    checksum: str
    accepted_checksum: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    status: ImageStatus
    created_at: datetime
    source_page_url: str | None = None
    license_name: str | None = None
    creator: str | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id cannot be empty")

        if not self.storage_uri.strip():
            raise ValueError("storage_uri cannot be empty")

        if not self.checksum.strip():
            raise ValueError("checksum cannot be empty")

        if not self.accepted_checksum.strip():
            raise ValueError("accepted_checksum cannot be empty")

        if self.width <= 0:
            raise ValueError("width must be greater than zero")

        if self.height <= 0:
            raise ValueError("height must be greater than zero")

        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be greater than zero")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    @classmethod
    def create(
        cls,
        *,
        image_id: str,
        source: ImageSource,
        storage_uri: str,
        checksum: str,
        accepted_checksum: str | None = None,
        content_type: str,
        width: int,
        height: int,
        size_bytes: int,
        status: ImageStatus = ImageStatus.ACCEPTED,
        source_page_url: str | None = None,
        license_name: str | None = None,
        creator: str | None = None,
    ) -> "ImageRecord":
        return cls(
            image_id=image_id,
            source=source,
            storage_uri=storage_uri,
            checksum=checksum,
            accepted_checksum=(accepted_checksum or checksum),
            content_type=content_type,
            width=width,
            height=height,
            size_bytes=size_bytes,
            status=status,
            created_at=datetime.now(UTC),
            source_page_url=source_page_url,
            license_name=license_name,
            creator=creator,
        )
