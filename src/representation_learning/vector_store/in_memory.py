"""In-memory vector search for local development and tests."""

import math
from collections.abc import Sequence

from representation_learning.vector_store.interface import (
    VectorRecord,
    VectorSearchResult,
)


class InMemoryVectorStore:
    def __init__(self, embedding_dimension: int) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        self._embedding_dimension = embedding_dimension
        self._records: dict[tuple[str, str], VectorRecord] = {}

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> None:
        for record in records:
            self._validate_embedding(record.embedding)

            key = (
                record.model_version,
                record.image_id,
            )
            self._records[key] = record

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        model_version: str,
        limit: int,
    ) -> tuple[VectorSearchResult, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        self._validate_embedding(query_embedding)

        results = [
            VectorSearchResult(
                image_id=record.image_id,
                storage_uri=record.storage_uri,
                score=self._cosine_similarity(
                    query_embedding,
                    record.embedding,
                ),
                model_version=record.model_version,
            )
            for record in self._records.values()
            if record.model_version == model_version
        ]

        results.sort(
            key=lambda result: (
                -result.score,
                result.image_id,
            )
        )

        return tuple(results[:limit])

    def delete(
        self,
        *,
        image_id: str,
        model_version: str,
    ) -> None:
        self._records.pop(
            (model_version, image_id),
            None,
        )

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> None:
        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                "Expected embedding dimension "
                f"{self._embedding_dimension}, "
                f"received {len(embedding)}"
            )

        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("Embedding contains a non-finite value")

    @staticmethod
    def _cosine_similarity(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(
                first,
                second,
                strict=True,
            )
        )
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))

        if first_norm == 0 or second_norm == 0:
            raise ValueError("Cannot compare a zero-length embedding")

        return dot_product / (first_norm * second_norm)
