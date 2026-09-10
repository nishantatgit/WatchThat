"""Generate image embeddings and upload them to the vector index."""

import argparse
from itertools import batched

from representation_learning.data.augmentations import (
    InferenceTransform,
)
from representation_learning.data.dataset import (
    MountedBlobImageReader,
    load_dataset_manifest,
)
from representation_learning.inference.embedder import (
    ImageEmbedder,
)
from representation_learning.inference.model_loader import (
    EncoderLoader,
)
from representation_learning.utils.config import (
    load_infrastructure_config,
    load_training_config,
)
from representation_learning.vector_store.azure_ai_search import (
    AzureAISearchVectorStore,
)
from representation_learning.vector_store.interface import (
    VectorRecord,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and index image embeddings",
    )
    parser.add_argument(
        "--manifest-path",
        required=True,
        help="Local path to manifest.jsonl",
    )
    parser.add_argument(
        "--image-mount-directory",
        required=True,
        help="Directory containing accepted images",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/best-checkpoint.pt",
        help="Trained model checkpoint",
    )
    parser.add_argument(
        "--index-name",
        default="image-embeddings-v1",
        help="Physical Azure AI Search index",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Unique version of the model producing the embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images embedded in one forward pass",
    )
    parser.add_argument(
        "--infrastructure-config",
        default="configs/azure.yaml",
    )
    parser.add_argument(
        "--training-config",
        default="configs/train.yaml",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    infrastructure = load_infrastructure_config(arguments.infrastructure_config)
    training = load_training_config(arguments.training_config)

    items = load_dataset_manifest(arguments.manifest_path)

    loaded = EncoderLoader().load(
        checkpoint_path=arguments.checkpoint_path,
        architecture_version=(training.model.architecture_version),
        projection_dimension=(training.model.projection_dimension),
    )

    if loaded.embedding_dimension != infrastructure.vector_store.embedding_dimension:
        raise ValueError("Model and vector-index embedding dimensions differ")

    reader = MountedBlobImageReader(arguments.image_mount_directory)

    embedder = ImageEmbedder(
        encoder=loaded.encoder,
        transform=InferenceTransform(image_size=training.data.image_size),
        device=loaded.device,
    )

    vector_store = AzureAISearchVectorStore(
        endpoint=infrastructure.vector_store.endpoint,
        index_name=arguments.index_name,
        embedding_dimension=loaded.embedding_dimension,
    )

    processed_count = 0

    for item_batch in batched(
        items,
        arguments.batch_size,
    ):
        images = [reader.read(item) for item in item_batch]
        embeddings = embedder.embed_batch(images)

        records = [
            VectorRecord(
                image_id=item.image_id,
                storage_uri=item.storage_uri,
                embedding=tuple(embedding.tolist()),
                model_version=arguments.model_version,
            )
            for item, embedding in zip(
                item_batch,
                embeddings,
                strict=True,
            )
        ]

        vector_store.upsert(records)

        processed_count += len(records)

        print(f"Indexed {processed_count}/{len(items)} images")

    print("Embedding backfill complete")
    print(
        "Training snapshot:",
        loaded.metadata.dataset_snapshot_id,
    )
    print("Model version:", arguments.model_version)
    print("Index:", arguments.index_name)


if __name__ == "__main__":
    main()
