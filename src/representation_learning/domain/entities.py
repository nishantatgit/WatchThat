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
from datetime import datetime, timezone
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
    content_type: str
    width: int
    height: int
    size_bytes: int
    status: ImageStatus
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id cannot be empty")

        if not self.storage_uri.strip():
            raise ValueError("storage_uri cannot be empty")

        if not self.checksum.strip():
            raise ValueError("checksum cannot be empty")

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
        content_type: str,
        width: int,
        height: int,
        size_bytes: int,
        status: ImageStatus = ImageStatus.ACCEPTED,
    ) -> "ImageRecord":
        return cls(
            image_id=image_id,
            source=source,
            storage_uri=storage_uri,
            checksum=checksum,
            content_type=content_type,
            width=width,
            height=height,
            size_bytes=size_bytes,
            status=status,
            created_at=datetime.now(timezone.utc),
        )
