"""YAML configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AzureSettings:
    subscription_id: str
    resource_group: str
    location: str


@dataclass(frozen=True, slots=True)
class StorageSettings:
    account_name: str
    account_url: str
    table_endpoint: str
    metadata_table: str
    raw_container: str
    quarantine_container: str
    accepted_container: str
    dataset_manifest_container: str


@dataclass(frozen=True, slots=True)
class MessagingSettings:
    fully_qualified_namespace: str
    ingestion_queue: str


@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    azure: AzureSettings
    storage: StorageSettings
    messaging: MessagingSettings


@dataclass(frozen=True, slots=True)
class ScrapingSettings:
    enabled: bool
    schedule: str
    quarantine_by_default: bool
    seed_urls: tuple[str, ...]
    allowed_page_hosts: frozenset[str]
    allowed_image_hosts: frozenset[str]
    maximum_pages: int
    maximum_depth: int
    maximum_images_per_run: int
    maximum_image_size_mb: int

    def __post_init__(self) -> None:
        if not self.seed_urls:
            raise ValueError("At least one scraper seed URL is required")

        if not self.allowed_page_hosts:
            raise ValueError("At least one page host must be allowed")

        if not self.allowed_image_hosts:
            raise ValueError("At least one image host must be allowed")

        if self.maximum_pages <= 0:
            raise ValueError("maximum_pages must be positive")

        if self.maximum_depth < 0:
            raise ValueError("maximum_depth cannot be negative")

        if self.maximum_images_per_run <= 0:
            raise ValueError("maximum_images_per_run must be positive")

        if self.maximum_image_size_mb <= 0:
            raise ValueError("maximum_image_size_mb must be positive")


@dataclass(frozen=True, slots=True)
class TrainingDataSettings:
    manifest_path: str | None
    image_mount_directory: str | None
    image_size: int
    batch_size: int
    num_workers: int
    split_seed: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        ratio_sum = self.train_ratio + self.validation_ratio + self.test_ratio

        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError("Dataset split ratios must add up to 1")


@dataclass(frozen=True, slots=True)
class TrainingModelSettings:
    input_channels: int
    embedding_dimension: int
    projection_dimension: int
    architecture_version: str


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    maximum_epochs: int
    learning_rate: float
    weight_decay: float
    temperature: float
    seed: int


@dataclass(frozen=True, slots=True)
class EarlyStoppingSettings:
    patience: int
    minimum_improvement: float


@dataclass(frozen=True, slots=True)
class CheckpointSettings:
    output_directory: str
    resume_from: str | None


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    project_name: str
    experiment_name: str
    data: TrainingDataSettings
    model: TrainingModelSettings
    training: OptimizationSettings
    early_stopping: EarlyStoppingSettings
    checkpointing: CheckpointSettings


def load_infrastructure_config(
    path: str | Path = "configs/azure.yaml",
) -> InfrastructureConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise TypeError("Configuration root must be a mapping")

    azure = _required_section(raw_config, "azure")
    storage = _required_section(raw_config, "storage")
    messaging = _required_section(raw_config, "messaging")

    return InfrastructureConfig(
        azure=AzureSettings(
            subscription_id=_required_string(azure, "subscription_id", "azure"),
            resource_group=_required_string(azure, "resource_group", "azure"),
            location=_required_string(azure, "location", "azure"),
        ),
        storage=StorageSettings(
            account_name=_required_string(storage, "account_name", "storage"),
            account_url=_required_string(storage, "account_url", "storage"),
            table_endpoint=_required_string(storage, "table_endpoint", "storage"),
            metadata_table=_required_string(storage, "metadata_table", "storage"),
            raw_container=_required_string(storage, "raw_container", "storage"),
            quarantine_container=_required_string(
                storage, "quarantine_container", "storage"
            ),
            accepted_container=_required_string(
                storage, "accepted_container", "storage"
            ),
            dataset_manifest_container=_required_string(
                storage,
                "dataset_manifest_container",
                "storage",
            ),
        ),
        messaging=MessagingSettings(
            fully_qualified_namespace=_required_string(
                messaging,
                "fully_qualified_namespace",
                "messaging",
            ),
            ingestion_queue=_required_string(
                messaging,
                "ingestion_queue",
                "messaging",
            ),
        ),
    )

def load_scraping_config(
    path: str | Path = "configs/ingestion.yaml",
) -> ScrapingSettings:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Ingestion configuration does not exist: {config_path}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise TypeError("Ingestion configuration root must be a mapping")

    scraping = _required_section(raw_config, "scraping")

    seed_urls = _required_string_tuple(
        scraping,
        "seed_urls",
        "scraping",
    )
    allowed_page_hosts = _required_string_tuple(
        scraping,
        "allowed_page_hosts",
        "scraping",
    )
    allowed_image_hosts = _required_string_tuple(
        scraping,
        "allowed_image_hosts",
        "scraping",
    )

    return ScrapingSettings(
        enabled=_required_bool(scraping, "enabled", "scraping"),
        schedule=_required_string(scraping, "schedule", "scraping"),
        quarantine_by_default=_required_bool(
            scraping,
            "quarantine_by_default",
            "scraping",
        ),
        seed_urls=seed_urls,
        allowed_page_hosts=frozenset(allowed_page_hosts),
        allowed_image_hosts=frozenset(allowed_image_hosts),
        maximum_pages=_required_int(
            scraping,
            "maximum_pages",
            "scraping",
        ),
        maximum_depth=_required_int(
            scraping,
            "maximum_depth",
            "scraping",
        ),
        maximum_images_per_run=_required_int(
            scraping,
            "maximum_images_per_run",
            "scraping",
        ),
        maximum_image_size_mb=_required_int(
            scraping,
            "maximum_image_size_mb",
            "scraping",
        ),
    )

def load_training_config(
    path: str | Path = "configs/train.yaml",
) -> TrainingRunConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise TypeError("Training configuration root must be a mapping")

    project = _required_section(raw_config, "project")
    data = _required_section(raw_config, "data")
    model = _required_section(raw_config, "model")
    training = _required_section(raw_config, "training")
    early_stopping = _required_section(
        raw_config,
        "early_stopping",
    )
    checkpointing = _required_section(
        raw_config,
        "checkpointing",
    )

    return TrainingRunConfig(
        project_name=_required_string(project, "name", "project"),
        experiment_name=_required_string(project, "experiment_name", "project"),
        data=TrainingDataSettings(
            manifest_path=_optional_string(data, "manifest_path", "data"),
            image_mount_directory=_optional_string(
                data, "image_mount_directory", "data"
            ),
            image_size=_required_int(data, "image_size", "data"),
            batch_size=_required_int(data, "batch_size", "data"),
            num_workers=_required_int(data, "num_workers", "data"),
            split_seed=_required_string(data, "split_seed", "data"),
            train_ratio=_required_float(data, "train_ratio", "data"),
            validation_ratio=_required_float(data, "validation_ratio", "data"),
            test_ratio=_required_float(data, "test_ratio", "data"),
        ),
        model=TrainingModelSettings(
            input_channels=_required_int(model, "input_channels", "model"),
            embedding_dimension=_required_int(model, "embedding_dimension", "model"),
            projection_dimension=_required_int(model, "projection_dimension", "model"),
            architecture_version=_required_string(
                model, "architecture_version", "model"
            ),
        ),
        training=OptimizationSettings(
            maximum_epochs=_required_int(training, "maximum_epochs", "training"),
            learning_rate=_required_float(training, "learning_rate", "training"),
            weight_decay=_required_float(training, "weight_decay", "training"),
            temperature=_required_float(training, "temperature", "training"),
            seed=_required_int(training, "seed", "training"),
        ),
        early_stopping=EarlyStoppingSettings(
            patience=_required_int(early_stopping, "patience", "early_stopping"),
            minimum_improvement=_required_float(
                early_stopping,
                "minimum_improvement",
                "early_stopping",
            ),
        ),
        checkpointing=CheckpointSettings(
            output_directory=_required_string(
                checkpointing,
                "output_directory",
                "checkpointing",
            ),
            resume_from=_optional_string(
                checkpointing,
                "resume_from",
                "checkpointing",
            ),
        ),
    )


def _required_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise TypeError(f"Missing or invalid configuration section: {section_name}")

    return section


def _required_string(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> str:
    value = section.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing configuration value: {section_name}.{key}")

    return value


def _optional_string(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> str | None:
    value = section.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{section_name}.{key} must be a string or null")

    return value


def _required_int(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> int:
    value = section.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be an integer")

    return value


def _required_float(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> float:
    value = section.get(key)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{section_name}.{key} must be numeric")

    return float(value)

def _required_bool(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> bool:
    value = section.get(key)

    if not isinstance(value, bool):
        raise TypeError(
            f"Configuration value {section_name}.{key} must be a boolean"
        )

    return value


def _required_string_tuple(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> tuple[str, ...]:
    value = section.get(key)

    if not isinstance(value, list) or not value:
        raise TypeError(
            f"Configuration value {section_name}.{key} "
            "must be a non-empty list"
        )

    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TypeError(
            f"Configuration value {section_name}.{key} "
            "must contain non-empty strings"
        )

    return tuple(item.strip() for item in value)
