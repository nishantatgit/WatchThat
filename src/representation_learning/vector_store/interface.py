"""Version-aware vector-store contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VectorRecord:
    image_id: str
    storage_uri: str
    embedding: tuple[float, ...]
    model_version: str


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    image_id: str
    storage_uri: str
    score: float
    model_version: str


class VectorStore(Protocol):
    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> None: ...

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        model_version: str,
        limit: int,
    ) -> tuple[VectorSearchResult, ...]: ...

    def delete(
        self,
        *,
        image_id: str,
        model_version: str,
    ) -> None: ...
