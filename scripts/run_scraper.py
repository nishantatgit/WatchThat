"""Daily scraper entry point."""

import argparse

import httpx

from representation_learning.scraper.crawler import (
    HtmlPageDownloader,
    HtmlPageParser,
    InMemoryUrlFrontier,
    RobotsPolicy,
    WebCrawler,
)
from representation_learning.scraper.image_downloader import (
    ImageDownloader,
    RawImagePublisher,
)
from representation_learning.storage.image_store import (
    AzureBlobImageStore,
    StorageArea,
)
from representation_learning.utils.config import (
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

    frontier = InMemoryUrlFrontier(
        maximum_urls=scraping_config.maximum_pages,
        maximum_depth=scraping_config.maximum_depth,
    )
    page_downloader = HtmlPageDownloader(
        allowed_hosts=scraping_config.allowed_page_hosts,
    )
    robots_policy = RobotsPolicy()
    crawler = WebCrawler(
        frontier=frontier,
        downloader=page_downloader,
        parser=HtmlPageParser(),
        robots_policy=robots_policy,
    )

    image_downloader = ImageDownloader(
        allowed_hosts=scraping_config.allowed_image_hosts,
        maximum_response_bytes=(scraping_config.maximum_image_size_mb * 1024 * 1024),
    )

    image_store = AzureBlobImageStore(
        account_url=infrastructure_config.storage.account_url,
        container_names={
            StorageArea.RAW: infrastructure_config.storage.raw_container,
            StorageArea.ACCEPTED: (infrastructure_config.storage.accepted_container),
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
        crawl_result = crawler.crawl(
            scraping_config.seed_urls,
        )

        candidates = crawl_result.images[: scraping_config.maximum_images_per_run]

        print(f"Pages downloaded: {crawl_result.pages_downloaded}")
        print(f"Pages blocked by robots: {crawl_result.pages_blocked_by_robots}")
        print(f"Image candidates: {len(crawl_result.images)}")
        print(f"Page failures: {len(crawl_result.failures)}")

        for candidate in candidates:
            try:
                downloaded = image_downloader.download(candidate)
            except (httpx.HTTPError, ValueError) as error:
                download_failure_count += 1
                print(f"Skipped image {candidate.image_url}: {error}")
                continue

            published = publisher.publish(
                candidate=candidate,
                downloaded=downloaded,
            )
            published_count += 1

            print(f"Published: {published.storage_uri}")
    finally:
        crawler.close()
        image_downloader.close()

    print(f"Images published: {published_count}")
    print(f"Image download failures: {download_failure_count}")


if __name__ == "__main__":
    main()
