"""Daily scraper entry point."""

import argparse

import httpx

from representation_learning.scraper.crawler import (
    HtmlPageDownloader,
    HtmlPageParser,
    InMemoryUrlFrontier,
    RobotsPolicy,
    ScrapedImageCandidate,
    WebCrawler,
)
from representation_learning.scraper.image_downloader import (
    ImageDownloader,
    RawImagePublisher,
)
from representation_learning.scraper.wikimedia import (
    WikimediaCommonsSource,
)
from representation_learning.storage.image_store import (
    AzureBlobImageStore,
    StorageArea,
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


def discover_wikimedia_images(
    config: ScrapingSettings,
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
        print(
            "Pages blocked by robots: "
            f"{result.pages_blocked_by_robots}"
        )
        print(f"Page failures: {len(result.failures)}")

        return result.images[: config.maximum_images_per_run]
    finally:
        crawler.close()


def discover_images(
    config: ScrapingSettings,
) -> tuple[ScrapedImageCandidate, ...]:
    if config.discovery_source == "wikimedia":
        return discover_wikimedia_images(config)

    if config.discovery_source == "generic_web":
        return discover_generic_web_images(config)

    raise ValueError(
        f"Unsupported discovery source: {config.discovery_source}"
    )


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

    candidates = discover_images(scraping_config)

    print(f"Discovery source: {scraping_config.discovery_source}")
    print(f"Image candidates: {len(candidates)}")

    image_downloader = ImageDownloader(
        allowed_hosts=scraping_config.allowed_image_hosts,
        maximum_response_bytes=(
            scraping_config.maximum_image_size_mb * 1024 * 1024
        ),
    )

    image_store = AzureBlobImageStore(
        account_url=infrastructure_config.storage.account_url,
        container_names={
            StorageArea.RAW: infrastructure_config.storage.raw_container,
            StorageArea.ACCEPTED: (
                infrastructure_config.storage.accepted_container
            ),
            StorageArea.QUARANTINE: (
                infrastructure_config.storage.quarantine_container
            ),
        },
    )
    publisher = RawImagePublisher(
        image_store=image_store,
    )

    published_count = 0
    download_failure_count = 0

    try:
        for candidate in candidates:
            try:
                downloaded = image_downloader.download(candidate)
            except (httpx.HTTPError, ValueError) as error:
                download_failure_count += 1
                print(f"Skipped {candidate.title}: {error}")
                continue

            published = publisher.publish(
                candidate=candidate,
                downloaded=downloaded,
            )
            published_count += 1

            print(f"Published: {published.storage_uri}")
    finally:
        image_downloader.close()

    print(f"Images published: {published_count}")
    print(f"Image download failures: {download_failure_count}")


if __name__ == "__main__":
    main()