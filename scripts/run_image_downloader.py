"""Consume scraper download requests and upload images to raw storage."""

import argparse
import json
import logging
from typing import Any

import httpx
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.servicebus import (
    ServiceBusClient,
    ServiceBusReceivedMessage,
)

from representation_learning.scraper.download_queue import (
    decode_download_message,
)
from representation_learning.scraper.image_downloader import (
    ImageDownloader,
    RawImagePublisher,
)
from representation_learning.scraper.source_policy import (
    ScrapingSourcePolicy,
)
from representation_learning.scraper.state_store import (
    AzureTableScraperStateStore,
    ScraperItemStatus,
    ScraperStateRecord,
)
from representation_learning.storage.image_store import (
    AzureBlobImageStore,
    StorageArea,
)
from representation_learning.utils.config import (
    load_infrastructure_config,
    load_scraping_config,
)

LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download queued scraper images",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/ingestion.yaml",
    )
    parser.add_argument(
        "--azure-config",
        default="configs/azure.yaml",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one available message and exit",
    )

    return parser.parse_args()


def process_message(
    *,
    message: ServiceBusReceivedMessage,
    receiver: Any,
    downloader: ImageDownloader,
    publisher: RawImagePublisher,
    source_policy: ScrapingSourcePolicy,
    state_store: AzureTableScraperStateStore,
    maximum_attempts: int,
) -> None:
    try:
        candidate = decode_download_message(message)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as error:
        LOGGER.exception("Dead-lettering malformed download message")

        receiver.dead_letter_message(
            message,
            reason="InvalidDownloadMessage",
            error_description=str(error)[:4096],
        )
        return

    record = state_store.get(candidate.source_page_url)

    if record is not None and record.status in {
        ScraperItemStatus.PUBLISHED,
        ScraperItemStatus.REJECTED,
    }:
        LOGGER.info(
            "Candidate already processed: %s",
            candidate.source_page_url,
        )
        receiver.complete_message(message)
        return

    if record is None:
        record = ScraperStateRecord.discovered(candidate)
        state_store.save(record)

    if record.attempt_count >= maximum_attempts:
        receiver.dead_letter_message(
            message,
            reason="MaximumAttemptsReached",
            error_description=(f"Candidate reached {record.attempt_count} attempts"),
        )
        return

    decision = source_policy.evaluate(candidate)

    if not decision.allowed:
        reason = decision.reason or "Rejected by policy"
        state_store.save(record.mark_rejected(reason))
        receiver.complete_message(message)

        LOGGER.warning(
            "Rejected candidate %s: %s",
            candidate.source_page_url,
            reason,
        )
        return

    try:
        downloaded = downloader.download(candidate)

        published = publisher.publish(
            candidate=candidate,
            downloaded=downloaded,
        )

        state_store.save(
            record.mark_published(published.storage_uri),
        )
        receiver.complete_message(message)

        LOGGER.info(
            "Published candidate to %s",
            published.storage_uri,
        )
    except ValueError as error:
        state_store.save(record.mark_rejected(str(error)))

        receiver.dead_letter_message(
            message,
            reason="InvalidImage",
            error_description=str(error)[:4096],
        )

        LOGGER.exception("Rejected invalid image")
    except (httpx.HTTPError, AzureError) as error:
        failed_record = record.mark_failed(str(error))
        state_store.save(failed_record)

        if failed_record.attempt_count >= maximum_attempts:
            receiver.dead_letter_message(
                message,
                reason="DownloadAttemptsExhausted",
                error_description=str(error)[:4096],
            )

            LOGGER.exception("Download attempts exhausted")
        else:
            receiver.abandon_message(message)

            LOGGER.exception("Download failed; message will be retried")
    except Exception:
        receiver.abandon_message(message)
        LOGGER.exception("Unexpected image-download failure")


def main() -> None:
    arguments = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scraping_config = load_scraping_config(
        arguments.ingestion_config,
    )
    infrastructure_config = load_infrastructure_config(
        arguments.azure_config,
    )
    credential = DefaultAzureCredential()

    state_store = AzureTableScraperStateStore(
        endpoint=infrastructure_config.storage.table_endpoint,
        table_name=infrastructure_config.storage.scraper_state_table,
        credential=credential,
    )

    image_store = AzureBlobImageStore(
        account_url=infrastructure_config.storage.account_url,
        credential=credential,
        container_names={
            StorageArea.RAW: infrastructure_config.storage.raw_container,
            StorageArea.ACCEPTED: (infrastructure_config.storage.accepted_container),
            StorageArea.QUARANTINE: (
                infrastructure_config.storage.quarantine_container
            ),
        },
    )

    downloader = ImageDownloader(
        allowed_hosts=scraping_config.allowed_image_hosts,
        maximum_response_bytes=(scraping_config.maximum_image_size_mb * 1024 * 1024),
    )
    publisher = RawImagePublisher(image_store=image_store)

    source_policy = ScrapingSourcePolicy(
        allowed_source_hosts=scraping_config.allowed_page_hosts,
        allowed_licenses=scraping_config.allowed_licenses,
        require_license=scraping_config.require_license,
    )

    service_bus = ServiceBusClient(
        fully_qualified_namespace=(
            infrastructure_config.messaging.fully_qualified_namespace
        ),
        credential=credential,
    )

    try:
        with service_bus:
            receiver = service_bus.get_queue_receiver(
                queue_name=(infrastructure_config.messaging.download_queue),
                max_wait_time=5,
            )

            with receiver:
                LOGGER.info(
                    "Listening to queue %s",
                    infrastructure_config.messaging.download_queue,
                )

                while True:
                    messages = receiver.receive_messages(
                        max_message_count=1,
                        max_wait_time=5,
                    )

                    for message in messages:
                        process_message(
                            message=message,
                            receiver=receiver,
                            downloader=downloader,
                            publisher=publisher,
                            source_policy=source_policy,
                            state_store=state_store,
                            maximum_attempts=(
                                scraping_config.maximum_candidate_attempts
                            ),
                        )

                    if arguments.once:
                        return
    finally:
        downloader.close()


if __name__ == "__main__":
    main()
