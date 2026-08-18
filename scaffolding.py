from pathlib import Path

PROJECT_ROOT = Path.cwd()

DIRECTORIES = [
    "configs",
    "data/raw",
    "data/quarantine",
    "data/accepted",
    "data/train",
    "data/validation",
    "artifacts",
    "outputs",
    "scripts",
    "azure/deployment",
    "azure/jobs",
    "tests",

    "src/representation_learning/data",
    "src/representation_learning/models",
    "src/representation_learning/losses",
    "src/representation_learning/training",
    "src/representation_learning/evaluation",
    "src/representation_learning/inference",

    "src/representation_learning/ingestion",
    "src/representation_learning/feature_store",
    "src/representation_learning/vector_store",
    "src/representation_learning/retraining",
    "src/representation_learning/scraper",
    "src/representation_learning/serving",
    "src/representation_learning/monitoring",
    "src/representation_learning/storage",
    "src/representation_learning/domain",
    "src/representation_learning/utils",
]

PACKAGE_DIRECTORIES = [
    "src/representation_learning",
    "src/representation_learning/data",
    "src/representation_learning/models",
    "src/representation_learning/losses",
    "src/representation_learning/training",
    "src/representation_learning/evaluation",
    "src/representation_learning/inference",
    "src/representation_learning/ingestion",
    "src/representation_learning/feature_store",
    "src/representation_learning/vector_store",
    "src/representation_learning/retraining",
    "src/representation_learning/scraper",
    "src/representation_learning/serving",
    "src/representation_learning/monitoring",
    "src/representation_learning/storage",
    "src/representation_learning/domain",
    "src/representation_learning/utils",
]

FILES = {
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    "configs/train.yaml": """\
project:
  name: image-representation-learning
  experiment_name: contrastive-learning

data:
  train_directory: data/train
  validation_directory: data/validation
  image_size: 224
  batch_size: 128
  num_workers: 4

model:
  input_channels: 3
  embedding_dimension: 256
  projection_dimension: 128
  architecture_version: cnn-v1

training:
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.0001
  temperature: 0.2
  seed: 42

checkpointing:
  output_directory: outputs
  save_every_epochs: 5
  resume_from: null
""",

    "configs/ingestion.yaml": """\
user_uploads:
  enabled: true
  process_immediately: true
  maximum_file_size_mb: 10

scraping:
  enabled: true
  schedule: "0 2 * * *"
  quarantine_by_default: true

validation:
  supported_formats:
    - jpg
    - jpeg
    - png
    - webp
  minimum_width: 128
  minimum_height: 128
  reject_corrupt_images: true
  exact_deduplication: true
  perceptual_deduplication: true
""",

    "configs/retraining.yaml": """\
evaluation_schedule: "0 4 * * *"

policy:
  maximum_interval_days: 7
  minimum_new_images: 10000
  minimum_dataset_growth_ratio: 0.10
  trigger_on_data_drift: true
  trigger_on_quality_regression: true

promotion:
  require_improvement: true
  require_no_metric_regression: true
  use_canary_deployment: true
  backfill_embeddings_before_promotion: true
""",

    "configs/serving.yaml": """\
api:
  host: "0.0.0.0"
  port: 8000
  maximum_batch_size: 32
  request_timeout_seconds: 30

latency_targets_ms:
  embedding_p95: 500
  vector_search_p95: 100
  end_to_end_p95: 700

model:
  embedding_dimension: 256
  active_version: null

vector_search:
  number_of_results: 10
  similarity_metric: cosine
""",

    "configs/azure.yaml": """\
azure:
  subscription_id: null
  resource_group: null
  location: northeurope

storage:
  account_name: null
  raw_container: raw-images
  quarantine_container: quarantined-images
  accepted_container: accepted-images

machine_learning:
  workspace_name: null
  compute_cluster_name: gpu-cluster
  vm_size: Standard_NC4as_T4_v3
  registered_model_name: image-encoder

vector_store:
  search_service_name: null
  active_index_alias: image-embeddings-active

feature_store:
  name: image-feature-store
  offline_store: adls
  online_store: managed_redis

deployment:
  container_registry_name: null
  container_app_environment: image-platform-environment
  inference_app_name: image-inference-api
""",

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    "src/representation_learning/domain/entities.py": '''\
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
''',

    "src/representation_learning/domain/events.py": '''\
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
''',

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    "src/representation_learning/storage/image_store.py": '''\
"""Image-storage abstraction.

Implementations will support:
- Local filesystem for development
- Azure Blob Storage for production
"""
''',

    "src/representation_learning/storage/metadata_store.py": '''\
"""Persistent image metadata and processing-status storage."""
''',

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    "src/representation_learning/data/dataset.py": '''\
"""PyTorch datasets for unlabelled image training."""
''',

    "src/representation_learning/data/augmentations.py": '''\
"""Stochastic training and deterministic inference transformations."""
''',

    "src/representation_learning/data/splitting.py": '''\
"""Leakage-safe training and validation dataset splitting."""
''',

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    "src/representation_learning/models/blocks.py": '''\
"""Custom CNN and residual building blocks using basic PyTorch layers."""
''',

    "src/representation_learning/models/encoder.py": '''\
"""Randomly initialized CNN image encoder.

No pretrained weights or imported model architecture will be used.
"""
''',

    "src/representation_learning/models/projection_head.py": '''\
"""Projection head used only during contrastive training."""
''',

    "src/representation_learning/models/contrastive_model.py": '''\
"""Composition of the image encoder and projection head."""
''',

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    "src/representation_learning/losses/contrastive_loss.py": '''\
"""NT-Xent loss implemented using PyTorch tensor operations."""
''',

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------
    "src/representation_learning/training/trainer.py": '''\
"""Training and validation loops."""
''',

    "src/representation_learning/training/checkpointing.py": '''\
"""Model, optimizer and scheduler checkpoint management."""
''',

    "src/representation_learning/training/experiment.py": '''\
"""MLflow experiment and model-artifact tracking."""
''',

    "src/representation_learning/evaluation/metrics.py": '''\
"""Representation-quality and embedding-collapse metrics."""
''',

    "src/representation_learning/evaluation/retrieval.py": '''\
"""Recall@K, precision@K and nearest-neighbour evaluation."""
''',

    "src/representation_learning/evaluation/model_comparison.py": '''\
"""Candidate-versus-production model evaluation."""
''',

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    "src/representation_learning/inference/embedder.py": '''\
"""Batch and single-image embedding generation."""
''',

    "src/representation_learning/inference/model_loader.py": '''\
"""Version-aware encoder loading and lifecycle management."""
''',

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    "src/representation_learning/ingestion/user_upload.py": '''\
"""Synchronous user-upload ingestion workflow."""
''',

    "src/representation_learning/ingestion/scraped_image.py": '''\
"""Quarantined ingestion workflow for scraped images."""
''',

    "src/representation_learning/ingestion/validation.py": '''\
"""Image format, dimensions, integrity and policy validation."""
''',

    "src/representation_learning/ingestion/deduplication.py": '''\
"""Exact hash and perceptual-hash image deduplication."""
''',

    "src/representation_learning/ingestion/event_handler.py": '''\
"""Idempotent handler for Blob Storage and Event Grid events."""
''',

    # ------------------------------------------------------------------
    # Feature store
    # ------------------------------------------------------------------
    "src/representation_learning/feature_store/interface.py": '''\
"""Feature-store contracts shared by local and Azure implementations."""
''',

    "src/representation_learning/feature_store/definitions.py": '''\
"""Versioned image-feature definitions."""
''',

    "src/representation_learning/feature_store/online_store.py": '''\
"""Low-latency online feature-store implementation."""
''',

    "src/representation_learning/feature_store/offline_store.py": '''\
"""Historical offline features for training and analysis."""
''',

    "src/representation_learning/feature_store/materialization.py": '''\
"""Offline and online feature-materialization workflows."""
''',

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    "src/representation_learning/vector_store/interface.py": '''\
"""Version-aware vector-store contract."""
''',

    "src/representation_learning/vector_store/azure_ai_search.py": '''\
"""Azure AI Search vector-index implementation."""
''',

    "src/representation_learning/vector_store/in_memory.py": '''\
"""In-memory vector search for local development and tests."""
''',

    "src/representation_learning/vector_store/index_management.py": '''\
"""Vector-index creation, backfill, alias switching and rollback."""
''',

    # ------------------------------------------------------------------
    # Scraper
    # ------------------------------------------------------------------
    "src/representation_learning/scraper/crawler.py": '''\
"""Daily image-web-scraping workflow."""
''',

    "src/representation_learning/scraper/source_policy.py": '''\
"""Source allowlists, robots rules, licensing and training eligibility."""
''',

    "src/representation_learning/scraper/scheduler.py": '''\
"""Scheduled scraper-job entry point."""
''',

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------
    "src/representation_learning/retraining/policy.py": '''\
"""Data-volume, elapsed-time, drift and quality retraining rules."""
''',

    "src/representation_learning/retraining/dataset_snapshot.py": '''\
"""Immutable and reproducible training dataset snapshots."""
''',

    "src/representation_learning/retraining/trigger.py": '''\
"""Azure ML retraining-job trigger."""
''',

    "src/representation_learning/retraining/embedding_backfill.py": '''\
"""Regenerate corpus embeddings for a candidate model version."""
''',

    "src/representation_learning/retraining/promotion.py": '''\
"""Canary validation, model promotion, index switching and rollback."""
''',

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    "src/representation_learning/serving/api.py": '''\
"""Real-time FastAPI application.

Planned endpoints:
- GET /health
- POST /images
- POST /embeddings
- POST /similar-images
"""
''',

    "src/representation_learning/serving/schemas.py": '''\
"""API request and response schemas."""
''',

    "src/representation_learning/serving/embedding_service.py": '''\
"""Orchestrates storage, embedding generation and vector indexing."""
''',

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    "src/representation_learning/monitoring/data_quality.py": '''\
"""Image-ingestion and feature-quality monitoring."""
''',

    "src/representation_learning/monitoring/drift.py": '''\
"""Image-feature and embedding-distribution drift detection."""
''',

    "src/representation_learning/monitoring/latency.py": '''\
"""Embedding, vector-search and end-to-end latency metrics."""
''',

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    "src/representation_learning/utils/config.py": '''\
"""YAML configuration loading and validation."""
''',

    "src/representation_learning/utils/logging.py": '''\
"""Structured application logging."""
''',

    "src/representation_learning/utils/reproducibility.py": '''\
"""Random-seed and deterministic-execution utilities."""
''',

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    "scripts/train.py": '''\
"""Training entry point."""


def main() -> None:
    raise NotImplementedError("Training has not been implemented yet.")


if __name__ == "__main__":
    main()
''',

    "scripts/evaluate.py": '''\
"""Representation-evaluation entry point."""


def main() -> None:
    raise NotImplementedError("Evaluation has not been implemented yet.")


if __name__ == "__main__":
    main()
''',

    "scripts/generate_embeddings.py": '''\
"""Batch embedding and vector-index backfill entry point."""


def main() -> None:
    raise NotImplementedError("Embedding generation has not been implemented yet.")


if __name__ == "__main__":
    main()
''',

    "scripts/run_scraper.py": '''\
"""Daily scraper entry point."""


def main() -> None:
    raise NotImplementedError("Web scraping has not been implemented yet.")


if __name__ == "__main__":
    main()
''',

    "scripts/evaluate_retraining.py": '''\
"""Evaluate whether a new training run should be started."""


def main() -> None:
    raise NotImplementedError("Retraining policy has not been implemented yet.")


if __name__ == "__main__":
    main()
''',

    # ------------------------------------------------------------------
    # Azure placeholders
    # ------------------------------------------------------------------
    "azure/environment.yaml": """\
# Azure ML training environment.
""",

    "azure/jobs/train-job.yaml": """\
# Azure ML encoder-training job.
""",

    "azure/jobs/scraper-job.yaml": """\
# Daily Azure Container Apps scraper job.
""",

    "azure/jobs/retraining-evaluator-job.yaml": """\
# Daily retraining-policy evaluation job.
""",

    "azure/deployment/inference-app.yaml": """\
# Real-time inference application deployment.
""",

    "azure/deployment/event-processor.yaml": """\
# Event-driven image-processing worker deployment.
""",

    "Dockerfile": """\
# Real-time inference container will be defined after local inference works.
""",

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    "tests/test_dataset.py": "",
    "tests/test_encoder.py": "",
    "tests/test_contrastive_loss.py": "",
    "tests/test_ingestion.py": "",
    "tests/test_feature_store.py": "",
    "tests/test_vector_store.py": "",
    "tests/test_retraining_policy.py": "",
    "tests/test_embedding_service.py": "",
    "tests/test_model_versioning.py": "",

    # ------------------------------------------------------------------
    # Empty tracked directories
    # ------------------------------------------------------------------
    "data/raw/.gitkeep": "",
    "data/quarantine/.gitkeep": "",
    "data/accepted/.gitkeep": "",
    "data/train/.gitkeep": "",
    "data/validation/.gitkeep": "",
    "artifacts/.gitkeep": "",
    "outputs/.gitkeep": "",
}


def create_scaffolding() -> None:
    """Create missing scaffolding without modifying existing files."""

    if not (PROJECT_ROOT / "pyproject.toml").exists():
        raise FileNotFoundError(
            "No pyproject.toml was found. "
            "Run this script from the root of your existing uv project."
        )

    created_directories = []
    created_files = []
    skipped_files = []

    for directory in DIRECTORIES:
        path = PROJECT_ROOT / directory

        if not path.exists():
            path.mkdir(parents=True)
            created_directories.append(directory)

    for package_directory in PACKAGE_DIRECTORIES:
        init_file = PROJECT_ROOT / package_directory / "__init__.py"

        if init_file.exists():
            skipped_files.append(str(init_file.relative_to(PROJECT_ROOT)))
            continue

        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.write_text("", encoding="utf-8")
        created_files.append(str(init_file.relative_to(PROJECT_ROOT)))

    for relative_path, content in FILES.items():
        destination = PROJECT_ROOT / relative_path

        if destination.exists():
            skipped_files.append(relative_path)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        created_files.append(relative_path)

    print(f"Project root: {PROJECT_ROOT.resolve()}")
    print(f"Created directories: {len(created_directories)}")
    print(f"Created files: {len(created_files)}")
    print(f"Skipped existing files: {len(skipped_files)}")

    if skipped_files:
        print("\nExisting files were preserved:")

        for path in skipped_files:
            print(f"  - {path}")


if __name__ == "__main__":
    create_scaffolding()