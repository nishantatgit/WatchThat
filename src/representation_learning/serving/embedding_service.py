"""Orchestrates embedding generation and vector search."""

from PIL import Image

from representation_learning.inference.embedder import (
    ImageEmbedder,
)
from representation_learning.serving.schemas import (
    SimilarImage,
)
from representation_learning.vector_store.interface import (
    VectorRecord,
    VectorStore,
)


class EmbeddingService:
    def __init__(
        self,
        *,
        embedder: ImageEmbedder,
        vector_store: VectorStore,
        model_version: str,
    ) -> None:
        if not model_version:
            raise ValueError("model_version cannot be empty")

        self._embedder = embedder
        self._vector_store = vector_store
        self._model_version = model_version

    def index_image(
        self,
        *,
        image_id: str,
        storage_uri: str,
        image: Image.Image,
    ) -> None:
        embedding = self._embedder.embed(image)

        record = VectorRecord(
            image_id=image_id,
            storage_uri=storage_uri,
            embedding=tuple(embedding.tolist()),
            model_version=self._model_version,
        )

        self._vector_store.upsert([record])

    def find_similar(
        self,
        *,
        image: Image.Image,
        limit: int,
    ) -> tuple[SimilarImage, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_embedding = self._embedder.embed(image)

        results = self._vector_store.search(
            query_embedding=query_embedding.tolist(),
            model_version=self._model_version,
            limit=limit,
        )

        return tuple(
            SimilarImage(
                image_id=result.image_id,
                storage_uri=result.storage_uri,
                similarity_score=result.score,
                model_version=result.model_version,
            )
            for result in results
        )
