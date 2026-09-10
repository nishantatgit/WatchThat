"""Azure AI Search vector-index creation and lifecycle management."""

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchAlias,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)


class AzureVectorIndexManager:
    _ALGORITHM_NAME = "image-vector-hnsw"
    _PROFILE_NAME = "image-vector-profile"

    def __init__(
        self,
        *,
        endpoint: str,
        credential: TokenCredential | None = None,
    ) -> None:
        self._client = SearchIndexClient(
            endpoint=endpoint,
            credential=credential or DefaultAzureCredential(),
        )

    def create_or_update(
        self,
        *,
        index_name: str,
        embedding_dimension: int,
    ) -> SearchIndex:
        if not index_name:
            raise ValueError("index_name cannot be empty")

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        fields = [
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="image_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="storage_uri",
                type=SearchFieldDataType.String,
            ),
            SimpleField(
                name="model_version",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                retrievable=False,
                vector_search_dimensions=embedding_dimension,
                vector_search_profile_name=self._PROFILE_NAME,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=self._ALGORITHM_NAME,
                    parameters=HnswParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name=self._PROFILE_NAME,
                    algorithm_configuration_name=(self._ALGORITHM_NAME),
                )
            ],
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )

        return self._client.create_or_update_index(index)

    def set_alias(
        self,
        *,
        alias_name: str,
        index_name: str,
    ) -> SearchAlias:
        if not alias_name:
            raise ValueError("alias_name cannot be empty")

        if not index_name:
            raise ValueError("index_name cannot be empty")

        alias = SearchAlias(
            name=alias_name,
            indexes=[index_name],
        )

        return self._client.create_or_update_alias(alias)
