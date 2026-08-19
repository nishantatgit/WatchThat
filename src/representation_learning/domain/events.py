"""Domain events.

Planned events:
- ImageUploaded
- ImageScraped
- ImageValidated
- ImageRejected
- FeaturesCalculated
- EmbeddingGenerated
- ImageIndexed
- RetrainingRequested
- ModelRegistered
- ModelPromoted
"""
# src/representation_learning/domain/events.py

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from representation_learning.domain.entities import ImageSource


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageReceived(DomainEvent):
    image_id: str
    source: ImageSource
    storage_uri: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageAccepted(DomainEvent):
    image_id: str
    storage_uri: str
    checksum: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageQuarantined(DomainEvent):
    image_id: str
    storage_uri: str
    reason: str
