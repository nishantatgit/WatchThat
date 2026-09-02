import argparse
import json
import logging
from collections.abc import Mapping
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.servicebus import (
    ServiceBusClient,
    ServiceBusReceivedMessage,
)

from representation_learning.ingestion.deduplication import (
    ImageDeduplicator,
)
from representation_learning.ingestion.event_handler import (
    BlobCreatedEventHandler,
)
from representation_learning.ingestion.image_optimizer import (
    ImageOptimizer,
)
from representation_learning.ingestion.validation import ImageValidator
from representation_learning.storage.image_store import (
    AzureBlobImageStore,
    StorageArea,
)
from representation_learning.storage.metadata_store import (
    AzureTableMetadataStore,
)
from representation_learning.utils.config import (
    load_infrastructure_config,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw-image events from Service Bus"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one available batch and exit",
    )
    return parser.parse_args()


def decode_event(
    message: ServiceBusReceivedMessage,
) -> dict[str, Any]:
    body = b"".join(bytes(part) for part in message.body)
    payload = json.loads(body)

    # Event Grid sometimes represents a delivered batch as a list.
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("Expected exactly one Event Grid event per message")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise TypeError("Event Grid message must contain an object")

    return payload


def extract_blob_uri(event: Mapping[str, Any]) -> str:
    event_type = event.get("eventType")

    if event_type != "Microsoft.Storage.BlobCreated":
        raise ValueError(f"Unsupported event type: {event_type}")

    data = event.get("data")

    if not isinstance(data, Mapping):
        raise TypeError("Event is missing its data object")

    blob_uri = data.get("url")

    if not isinstance(blob_uri, str) or not blob_uri:
        raise ValueError("Blob-created event is missing data.url")

    return blob_uri


def build_handler() -> tuple[
    BlobCreatedEventHandler,
    ServiceBusClient,
    str,
]:
    config = load_infrastructure_config()
    credential = DefaultAzureCredential()

    metadata_store = AzureTableMetadataStore(
        endpoint=config.storage.table_endpoint,
        table_name=config.storage.metadata_table,
        credential=credential,
    )

    image_store = AzureBlobImageStore(
        account_url=config.storage.account_url,
        credential=credential,
        container_names={
            StorageArea.RAW: config.storage.raw_container,
            StorageArea.ACCEPTED: config.storage.accepted_container,
            StorageArea.QUARANTINE: config.storage.quarantine_container,
        },
    )

    handler = BlobCreatedEventHandler(
        validator=ImageValidator(),
        deduplicator=ImageDeduplicator(metadata_store),
        optimizer=ImageOptimizer(
            maximum_dimension=1024,
            jpeg_quality=85,
        ),
        image_store=image_store,
        metadata_store=metadata_store,
    )

    service_bus = ServiceBusClient(
        fully_qualified_namespace=(config.messaging.fully_qualified_namespace),
        credential=credential,
    )

    return handler, service_bus, config.messaging.ingestion_queue


def process_message(
    *,
    message: ServiceBusReceivedMessage,
    receiver: Any,
    handler: BlobCreatedEventHandler,
) -> None:
    LOGGER.info(
        "Received message_id=%s delivery_count=%s",
        message.message_id,
        message.delivery_count,
    )
    try:
        event = decode_event(message)
        raw_blob_uri = extract_blob_uri(event)

        LOGGER.info("Processing raw blob: %s", raw_blob_uri)

        result = handler.handle(raw_blob_uri=raw_blob_uri)

        receiver.complete_message(message)

        LOGGER.info(
            "Processed image_id=%s accepted=%s quarantined=%s duplicate=%s",
            result.image_id,
            result.accepted,
            result.quarantined,
            result.is_duplicate,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        LOGGER.exception("Dead-lettering malformed message")

        receiver.dead_letter_message(
            message,
            reason="InvalidEvent",
            error_description=str(error)[:4096],
        )
    except Exception:
        # The message returns to the queue and can be retried.
        LOGGER.exception("Image processing failed; abandoning message")
        receiver.abandon_message(message)


def main() -> None:
    arguments = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    handler, service_bus, queue_name = build_handler()

    with service_bus:
        receiver = service_bus.get_queue_receiver(
            queue_name=queue_name,
            max_wait_time=5,
        )

        with receiver:
            LOGGER.info("Listening to queue %s", queue_name)

            while True:
                messages = receiver.receive_messages(
                    max_message_count=1,
                    max_wait_time=5,
                )
                for message in messages:
                    process_message(
                        message=message,
                        receiver=receiver,
                        handler=handler,
                    )

                if arguments.once:
                    return


if __name__ == "__main__":
    main()
