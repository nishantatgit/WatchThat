from io import BytesIO
from pathlib import Path

from PIL import Image

from representation_learning.ingestion.deduplication import (
    ImageDeduplicator,
    InMemoryChecksumRegistry,
)
from representation_learning.ingestion.event_handler import (
    BlobCreatedEventHandler,
)
from representation_learning.ingestion.validation import ImageValidator
from representation_learning.storage.image_store import (
    LocalImageStore,
    StorageArea,
)
from representation_learning.storage.metadata_store import (
    InMemoryMetadataStore,
)


def create_jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(
        buffer,
        format="JPEG",
    )
    return buffer.getvalue()


def test_valid_raw_image_is_accepted(tmp_path: Path) -> None:
    image_store = LocalImageStore(tmp_path)
    metadata_store = InMemoryMetadataStore()

    handler = BlobCreatedEventHandler(
        validator=ImageValidator(),
        deduplicator=ImageDeduplicator(
            InMemoryChecksumRegistry()
        ),
        image_store=image_store,
        metadata_store=metadata_store,
    )

    raw_uri = image_store.save(
        image_id="image-001",
        content=create_jpeg_bytes(),
        area=StorageArea.RAW,
        extension="jpg",
    )

    result = handler.handle(raw_blob_uri=raw_uri)

    assert result.accepted
    assert not result.quarantined
    assert not result.is_duplicate
    assert result.record is not None
    assert result.record.width == 100
    assert result.record.height == 100

    assert not Path(raw_uri).exists()
    assert Path(result.record.storage_uri).exists()
    assert metadata_store.get("image-001") == result.record


def test_invalid_raw_image_is_quarantined(tmp_path: Path) -> None:
    image_store = LocalImageStore(tmp_path)

    handler = BlobCreatedEventHandler(
        validator=ImageValidator(),
        deduplicator=ImageDeduplicator(
            InMemoryChecksumRegistry()
        ),
        image_store=image_store,
        metadata_store=InMemoryMetadataStore(),
    )

    raw_uri = image_store.save(
        image_id="image-002",
        content=b"not a valid image",
        area=StorageArea.RAW,
        extension="jpg",
    )

    result = handler.handle(raw_blob_uri=raw_uri)

    assert not result.accepted
    assert result.quarantined
    assert not result.is_duplicate
    assert result.record is None
    assert not Path(raw_uri).exists()

    quarantined_event = result.events[-1]
    assert Path(quarantined_event.storage_uri).exists()


def test_duplicate_raw_image_is_not_stored_twice(
    tmp_path: Path,
) -> None:
    image_store = LocalImageStore(tmp_path)
    metadata_store = InMemoryMetadataStore()
    checksum_registry = InMemoryChecksumRegistry()

    handler = BlobCreatedEventHandler(
        validator=ImageValidator(),
        deduplicator=ImageDeduplicator(checksum_registry),
        image_store=image_store,
        metadata_store=metadata_store,
    )

    content = create_jpeg_bytes()

    first_raw_uri = image_store.save(
        image_id="image-001",
        content=content,
        area=StorageArea.RAW,
        extension="jpg",
    )
    first_result = handler.handle(raw_blob_uri=first_raw_uri)

    second_raw_uri = image_store.save(
        image_id="image-002",
        content=content,
        area=StorageArea.RAW,
        extension="jpg",
    )
    second_result = handler.handle(raw_blob_uri=second_raw_uri)

    assert first_result.accepted
    assert second_result.is_duplicate
    assert second_result.duplicate_of == "image-001"
    assert not Path(second_raw_uri).exists()
    assert len(metadata_store.list_all()) == 1