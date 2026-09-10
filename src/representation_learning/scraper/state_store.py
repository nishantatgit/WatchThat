"""Persistent state for discovered scraper candidates."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Protocol

from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableClient, UpdateMode
from azure.identity import DefaultAzureCredential

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)


class ScraperItemStatus(StrEnum):
    DISCOVERED = "discovered"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ScraperStateRecord:
    source_page_url: str
    image_url: str
    status: ScraperItemStatus
    attempt_count: int
    last_error: str | None
    storage_uri: str | None
    discovered_at: datetime
    updated_at: datetime

    @classmethod
    def discovered(
        cls,
        candidate: ScrapedImageCandidate,
    ) -> "ScraperStateRecord":
        now = datetime.now(UTC)

        return cls(
            source_page_url=candidate.source_page_url,
            image_url=candidate.image_url,
            status=ScraperItemStatus.DISCOVERED,
            attempt_count=0,
            last_error=None,
            storage_uri=None,
            discovered_at=now,
            updated_at=now,
        )

    def mark_published(
        self,
        storage_uri: str,
    ) -> "ScraperStateRecord":
        return replace(
            self,
            status=ScraperItemStatus.PUBLISHED,
            attempt_count=self.attempt_count + 1,
            last_error=None,
            storage_uri=storage_uri,
            updated_at=datetime.now(UTC),
        )

    def mark_rejected(
        self,
        reason: str,
    ) -> "ScraperStateRecord":
        return replace(
            self,
            status=ScraperItemStatus.REJECTED,
            last_error=reason,
            updated_at=datetime.now(UTC),
        )

    def mark_failed(
        self,
        reason: str,
    ) -> "ScraperStateRecord":
        return replace(
            self,
            status=ScraperItemStatus.FAILED,
            attempt_count=self.attempt_count + 1,
            last_error=reason,
            updated_at=datetime.now(UTC),
        )


class ScraperStateStore(Protocol):
    def get(
        self,
        source_page_url: str,
    ) -> ScraperStateRecord | None: ...

    def save(
        self,
        record: ScraperStateRecord,
    ) -> None: ...


class InMemoryScraperStateStore:
    def __init__(self) -> None:
        self._records: dict[str, ScraperStateRecord] = {}

    def get(
        self,
        source_page_url: str,
    ) -> ScraperStateRecord | None:
        return self._records.get(source_page_url)

    def save(
        self,
        record: ScraperStateRecord,
    ) -> None:
        self._records[record.source_page_url] = record


class AzureTableScraperStateStore:
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

    def get(
        self,
        source_page_url: str,
    ) -> ScraperStateRecord | None:
        key = self._key(source_page_url)

        try:
            entity = self._table.get_entity(
                partition_key=key[:2],
                row_key=key,
            )
        except ResourceNotFoundError:
            return None

        discovered_at = entity["DiscoveredAt"]
        updated_at = entity["UpdatedAt"]

        if isinstance(discovered_at, str):
            discovered_at = datetime.fromisoformat(discovered_at)

        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return ScraperStateRecord(
            source_page_url=entity["SourcePageUrl"],
            image_url=entity["ImageUrl"],
            status=ScraperItemStatus(entity["Status"]),
            attempt_count=int(entity["AttemptCount"]),
            last_error=entity.get("LastError"),
            storage_uri=entity.get("StorageUri"),
            discovered_at=discovered_at,
            updated_at=updated_at,
        )

    def save(
        self,
        record: ScraperStateRecord,
    ) -> None:
        key = self._key(record.source_page_url)

        entity = {
            "PartitionKey": key[:2],
            "RowKey": key,
            "SourcePageUrl": record.source_page_url,
            "ImageUrl": record.image_url,
            "Status": record.status.value,
            "AttemptCount": record.attempt_count,
            "DiscoveredAt": record.discovered_at,
            "UpdatedAt": record.updated_at,
        }

        if record.last_error is not None:
            entity["LastError"] = record.last_error[:4096]

        if record.storage_uri is not None:
            entity["StorageUri"] = record.storage_uri

        self._table.upsert_entity(
            entity=entity,
            mode=UpdateMode.REPLACE,
        )

    @staticmethod
    def _key(source_page_url: str) -> str:
        return sha256(source_page_url.encode("utf-8")).hexdigest()
