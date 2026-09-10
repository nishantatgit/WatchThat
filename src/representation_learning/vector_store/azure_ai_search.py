"""Azure AI Search vector-store implementation."""

import math
from collections.abc import Sequence
from hashlib import sha256
from itertools import batched

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from representation_learning.vector_store.interface import (
    VectorRecord,
    VectorSearchResult,
)


class AzureAISearchVectorStore:
    _UPLOAD_BATCH_SIZE = 500

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        embedding_dimension: int,
        credential: TokenCredential | None = None,
    ) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        self._client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential or DefaultAzureCredential(),
        )
        self._embedding_dimension = embedding_dimension

    def upsert(
        self,
        records: Sequence[VectorRecord],
    ) -> None:
        documents = []

        for record in records:
            self._validate_embedding(record.embedding)

            documents.append(
                {
                    "document_id": self._document_id(
                        image_id=record.image_id,
                        model_version=record.model_version,
                    ),
                    "image_id": record.image_id,
                    "storage_uri": record.storage_uri,
                    "model_version": record.model_version,
                    "embedding": list(record.embedding),
                }
            )

        for batch in batched(
            documents,
            self._UPLOAD_BATCH_SIZE,
        ):
            results = self._client.upload_documents(
                documents=list(batch),
            )

            failures = [result for result in results if not result.succeeded]

            if failures:
                messages = [
                    f"{failure.key}: {failure.error_message}" for failure in failures
                ]

                raise RuntimeError(
                    "Failed to upload vector documents: " + "; ".join(messages)
                )

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

        vector_query = VectorizedQuery(
            vector=list(query_embedding),
            k_nearest_neighbors=limit,
            fields="embedding",
        )

        escaped_model_version = model_version.replace(
            "'",
            "''",
        )

        results = self._client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=(f"model_version eq '{escaped_model_version}'"),
            select=[
                "image_id",
                "storage_uri",
                "model_version",
            ],
            top=limit,
        )

        return tuple(
            VectorSearchResult(
                image_id=result["image_id"],
                storage_uri=result["storage_uri"],
                score=float(result["@search.score"]),
                model_version=result["model_version"],
            )
            for result in results
        )

    def delete(
        self,
        *,
        image_id: str,
        model_version: str,
    ) -> None:
        document_id = self._document_id(
            image_id=image_id,
            model_version=model_version,
        )

        results = self._client.delete_documents(
            documents=[
                {
                    "document_id": document_id,
                }
            ]
        )

        failures = [result for result in results if not result.succeeded]

        if failures:
            raise RuntimeError(
                f"Failed to delete vector document: {failures[0].error_message}"
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
    def _document_id(
        *,
        image_id: str,
        model_version: str,
    ) -> str:
        value = f"{model_version}\0{image_id}"
        return sha256(value.encode("utf-8")).hexdigest()
