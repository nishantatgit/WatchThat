"""Wikimedia Commons image discovery through the MediaWiki API."""

from html import unescape
from typing import Any

import httpx
from bs4 import BeautifulSoup

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)


class WikimediaCommonsSource:
    def __init__(
        self,
        *,
        api_url: str = "https://commons.wikimedia.org/w/api.php",
        timeout_seconds: float = 20.0,
        user_agent: str = "RepresentationLearningCrawler/0.1",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._api_url = api_url
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )

    def discover(
        self,
        *,
        category: str,
        limit: int,
    ) -> tuple[ScrapedImageCandidate, ...]:
        if not category.strip():
            raise ValueError("category cannot be empty")

        if limit <= 0:
            raise ValueError("limit must be positive")

        category_title = category.strip()

        if not category_title.casefold().startswith("category:"):
            category_title = f"Category:{category_title}"

        response = self._client.get(
            self._api_url,
            params={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "categorymembers",
                "gcmtitle": category_title,
                "gcmtype": "file",
                "gcmlimit": min(limit, 500),
                "prop": "info|imageinfo",
                "inprop": "url",
                "iiprop": "url|mime|size|extmetadata",
            },
        )
        response.raise_for_status()

        payload = response.json()
        pages = payload.get("query", {}).get("pages", [])

        if not isinstance(pages, list):
            raise TypeError("Wikimedia response pages must be a list")

        candidates: list[ScrapedImageCandidate] = []

        for page in pages:
            candidate = self._to_candidate(page)

            if candidate is not None:
                candidates.append(candidate)

            if len(candidates) >= limit:
                break

        return tuple(candidates)

    def close(self) -> None:
        self._client.close()

    @classmethod
    def _to_candidate(
        cls,
        page: dict[str, Any],
    ) -> ScrapedImageCandidate | None:
        image_info_items = page.get("imageinfo")

        if not isinstance(image_info_items, list) or not image_info_items:
            return None

        image_info = image_info_items[0]

        if not isinstance(image_info, dict):
            return None

        image_url = image_info.get("url")
        source_page_url = page.get("canonicalurl")
        mime_type = image_info.get("mime")

        if not isinstance(image_url, str):
            return None

        if not isinstance(source_page_url, str):
            return None

        if mime_type not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            return None

        metadata = image_info.get("extmetadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return ScrapedImageCandidate(
            image_url=image_url,
            source_page_url=source_page_url,
            license_name=cls._metadata_text(
                metadata,
                "LicenseShortName",
            ),
            creator=cls._metadata_text(
                metadata,
                "Artist",
            ),
            title=cls._metadata_text(
                metadata,
                "ObjectName",
            )
            or cls._page_title(page),
        )

    @staticmethod
    def _metadata_text(
        metadata: dict[str, Any],
        key: str,
    ) -> str | None:
        item = metadata.get(key)

        if not isinstance(item, dict):
            return None

        value = item.get("value")

        if not isinstance(value, str) or not value.strip():
            return None

        text = BeautifulSoup(
            unescape(value),
            "html.parser",
        ).get_text(
            " ",
            strip=True,
        )

        return text or None

    @staticmethod
    def _page_title(
        page: dict[str, Any],
    ) -> str | None:
        title = page.get("title")

        if not isinstance(title, str):
            return None

        if title.startswith("File:"):
            return title.removeprefix("File:")

        return title
