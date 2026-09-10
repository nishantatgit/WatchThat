"""Daily scraper entry point."""

import argparse
from collections.abc import Callable

from azure.core.exceptions import AzureError

from representation_learning.scraper.crawler import (
    HtmlPageDownloader,
    HtmlPageParser,
    InMemoryUrlFrontier,
    RobotsPolicy,
    ScrapedImageCandidate,
    WebCrawler,
)
from representation_learning.scraper.download_queue import (
    ImageDownloadQueuePublisher,
)
from representation_learning.scraper.source_policy import (
    ScrapingSourcePolicy,
)
from representation_learning.scraper.state_store import (
    AzureTableScraperStateStore,
    ScraperItemStatus,
    ScraperStateRecord,
    ScraperStateStore,
)
from representation_learning.scraper.wikimedia import (
    WikimediaCommonsSource,
)
from representation_learning.utils.config import (
    ScrapingSettings,
    load_infrastructure_config,
    load_scraping_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover web images and publish them to raw storage",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/ingestion.yaml",
    )
    parser.add_argument(
        "--azure-config",
        default="configs/azure.yaml",
    )

    return parser.parse_args()


def should_process_candidate(
    *,
    candidate: ScrapedImageCandidate,
    state_store: ScraperStateStore,
    maximum_attempts: int,
) -> bool:
    record = state_store.get(candidate.source_page_url)

    if record is None:
        return True

    if record.status in {
        ScraperItemStatus.QUEUED,
        ScraperItemStatus.PUBLISHED,
        ScraperItemStatus.REJECTED,
    }:
        return False

    return record.attempt_count < maximum_attempts


def discover_wikimedia_images(
    config: ScrapingSettings,
    should_include: Callable[
        [ScrapedImageCandidate],
        bool,
    ],
) -> tuple[ScrapedImageCandidate, ...]:
    source = WikimediaCommonsSource()
    candidates: dict[str, ScrapedImageCandidate] = {}

    try:
        for category in config.wikimedia_categories:
            remaining = config.maximum_images_per_run - len(candidates)

            if remaining <= 0:
                break

            discovered = source.discover(
                category=category,
                limit=remaining,
                maximum_category_depth=(config.maximum_category_depth),
                maximum_categories=config.maximum_categories,
                should_include=should_include,
            )

            for candidate in discovered:
                candidates.setdefault(
                    candidate.image_url,
                    candidate,
                )
    finally:
        source.close()

    return tuple(candidates.values())


def discover_generic_web_images(
    config: ScrapingSettings,
    should_include: Callable[
        [ScrapedImageCandidate],
        bool,
    ],
) -> tuple[ScrapedImageCandidate, ...]:
    frontier = InMemoryUrlFrontier(
        maximum_urls=config.maximum_pages,
        maximum_depth=config.maximum_depth,
    )
    crawler = WebCrawler(
        frontier=frontier,
        downloader=HtmlPageDownloader(
            allowed_hosts=config.allowed_page_hosts,
        ),
        parser=HtmlPageParser(),
        robots_policy=RobotsPolicy(),
    )

    try:
        result = crawler.crawl(config.seed_urls)

        print(f"Pages downloaded: {result.pages_downloaded}")
        print(f"Pages blocked by robots: {result.pages_blocked_by_robots}")
        print(f"Page failures: {len(result.failures)}")

        return tuple(
            candidate for candidate in result.images if should_include(candidate)
        )[: config.maximum_images_per_run]
    finally:
        crawler.close()


def discover_images(
    config: ScrapingSettings,
    should_include: Callable[
        [ScrapedImageCandidate],
        bool,
    ],
) -> tuple[ScrapedImageCandidate, ...]:
    if config.discovery_source == "wikimedia":
        return discover_wikimedia_images(config, should_include)

    if config.discovery_source == "generic_web":
        return discover_generic_web_images(config, should_include)

    raise ValueError(f"Unsupported discovery source: {config.discovery_source}")


def main() -> None:
    arguments = parse_arguments()

    scraping_config = load_scraping_config(
        arguments.ingestion_config,
    )
    infrastructure_config = load_infrastructure_config(
        arguments.azure_config,
    )

    if not scraping_config.enabled:
        print("Scraping is disabled")
        return

    state_store = AzureTableScraperStateStore(
        endpoint=infrastructure_config.storage.table_endpoint,
        table_name=(infrastructure_config.storage.scraper_state_table),
    )

    candidates = discover_images(
        scraping_config,
        lambda candidate: should_process_candidate(
            candidate=candidate,
            state_store=state_store,
            maximum_attempts=(scraping_config.maximum_candidate_attempts),
        ),
    )

    print(f"Discovery source: {scraping_config.discovery_source}")
    print(f"Image candidates: {len(candidates)}")

    source_policy = ScrapingSourcePolicy(
        allowed_source_hosts=scraping_config.allowed_page_hosts,
        allowed_licenses=scraping_config.allowed_licenses,
        require_license=scraping_config.require_license,
    )

    queue_publisher = ImageDownloadQueuePublisher(
        fully_qualified_namespace=(
            infrastructure_config.messaging.fully_qualified_namespace
        ),
        queue_name=infrastructure_config.messaging.download_queue,
    )

    queued_count = 0
    policy_rejection_count = 0
    queue_failure_count = 0

    try:
        for candidate in candidates:
            existing_record = state_store.get(
                candidate.source_page_url,
            )

            if existing_record is not None and existing_record.status in {
                ScraperItemStatus.QUEUED,
                ScraperItemStatus.PUBLISHED,
                ScraperItemStatus.REJECTED,
            }:
                print(f"Already processed: {candidate.title}")
                continue

            if (
                existing_record is not None
                and existing_record.attempt_count
                >= scraping_config.maximum_candidate_attempts
            ):
                print(f"Maximum attempts reached: {candidate.title}")
                continue

            record = existing_record or ScraperStateRecord.discovered(
                candidate,
            )

            if existing_record is None:
                state_store.save(record)

            decision = source_policy.evaluate(candidate)

            if not decision.allowed:
                policy_rejection_count += 1
                reason = decision.reason or "Rejected by policy"

                state_store.save(record.mark_rejected(reason))

                print(f"Rejected {candidate.title}: {reason}")
                continue

            try:
                message_id = queue_publisher.publish(candidate)
            except AzureError as error:
                queue_failure_count += 1
                state_store.save(record.mark_failed(str(error)))

                print(f"Failed to queue {candidate.title}: {error}")
                continue

            latest_record = state_store.get(candidate.source_page_url) or record

            if latest_record.status not in {
                ScraperItemStatus.PUBLISHED,
                ScraperItemStatus.REJECTED,
            }:
                state_store.save(latest_record.mark_queued())

            queued_count += 1

            print(f"Queued: {candidate.title} (message_id={message_id})")
    finally:
        queue_publisher.close()

    print(f"Images queued: {queued_count}")
    print(f"Policy rejections: {policy_rejection_count}")
    print(f"Queue failures: {queue_failure_count}")


if __name__ == "__main__":
    main()
