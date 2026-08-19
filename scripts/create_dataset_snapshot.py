"""Create and persist an immutable training dataset snapshot."""

from azure.identity import DefaultAzureCredential

from representation_learning.retraining.dataset_snapshot import (
    AzureBlobDatasetSnapshotStore,
    DatasetSnapshotBuilder,
)
from representation_learning.storage.metadata_store import (
    AzureTableMetadataStore,
)
from representation_learning.utils.config import (
    load_infrastructure_config,
)


def main() -> None:
    config = load_infrastructure_config()
    credential = DefaultAzureCredential()

    metadata_store = AzureTableMetadataStore(
        endpoint=config.storage.table_endpoint,
        table_name=config.storage.metadata_table,
        credential=credential,
    )

    snapshot_builder = DatasetSnapshotBuilder(
        metadata_store=metadata_store,
    )

    snapshot_store = AzureBlobDatasetSnapshotStore(
        account_url=config.storage.account_url,
        container_name=config.storage.dataset_manifest_container,
        credential=credential,
    )

    snapshot = snapshot_builder.build()
    manifest_uri = snapshot_store.save(snapshot)

    print(f"Snapshot ID: {snapshot.snapshot_id}")
    print(f"Image count: {snapshot.image_count}")
    print(f"Manifest URI: {manifest_uri}")


if __name__ == "__main__":
    main()