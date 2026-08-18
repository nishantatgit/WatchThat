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

@dataclass(frozen=True, slots=True)
class MessagingSettings:
    fully_qualified_namespace: str
    ingestion_queue: str

@dataclass(frozen=True, slots=True)
class InfrastructureConfig:
    azure: AzureSettings
    storage: StorageSettings
    messaging: MessagingSettings




def load_infrastructure_config(
    path: str | Path = "configs/azure.yaml",
) -> InfrastructureConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration root must be a mapping")

    azure = _required_section(raw_config, "azure")
    storage = _required_section(raw_config, "storage")
    messaging = _required_section(raw_config, "messaging")


    return InfrastructureConfig(
        azure=AzureSettings(
            subscription_id=_required_string(
                azure, "subscription_id", "azure"
            ),
            resource_group=_required_string(
                azure, "resource_group", "azure"
            ),
            location=_required_string(azure, "location", "azure"),
        ),
        storage=StorageSettings(
            account_name=_required_string(
                storage, "account_name", "storage"
            ),
            account_url=_required_string(
                storage, "account_url", "storage"
            ),
            table_endpoint=_required_string(
                storage, "table_endpoint", "storage"
            ),
            metadata_table=_required_string(
                storage, "metadata_table", "storage"
            ),
            raw_container=_required_string(
                storage, "raw_container", "storage"
            ),
            quarantine_container=_required_string(
                storage, "quarantine_container", "storage"
            ),
            accepted_container=_required_string(
                storage, "accepted_container", "storage"
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


def _required_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Missing or invalid configuration section: {section_name}"
        )

    return section


def _required_string(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> str:
    value = section.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Missing configuration value: {section_name}.{key}"
        )

    return value