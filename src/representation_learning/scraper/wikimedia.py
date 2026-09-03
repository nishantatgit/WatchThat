"""Wikimedia Commons image discovery through the MediaWiki API."""

from collections import deque
from dataclasses import dataclass
from html import unescape
from typing import Any

import httpx
from bs4 import BeautifulSoup

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)


@dataclass(frozen=True, slots=True)
class WikimediaCategoryRequest:
    title: str
    depth: int


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
        maximum_category_depth: int = 0,
        maximum_categories: int = 1,
    ) -> tuple[ScrapedImageCandidate, ...]:
        if not category.strip():
            raise ValueError("category cannot be empty")

        if limit <= 0:
            raise ValueError("limit must be positive")

        if maximum_category_depth < 0:
            raise ValueError("maximum_category_depth cannot be negative")

        if maximum_categories <= 0:
            raise ValueError("maximum_categories must be positive")

        initial_category = self._normalize_category(category)

        pending = deque(
            [
                WikimediaCategoryRequest(
                    title=initial_category,
                    depth=0,
                )
            ]
        )
        queued_categories = {initial_category.casefold()}
        visited_categories: set[str] = set()
        candidates: dict[str, ScrapedImageCandidate] = {}

        while (
            pending
            and len(visited_categories) < maximum_categories
            and len(candidates) < limit
        ):
            request = pending.popleft()
            category_key = request.title.casefold()

            if category_key in visited_categories:
                continue

            visited_categories.add(category_key)
            continuation: str | None = None

            while len(candidates) < limit:
                payload = self._query_category(
                    category=request.title,
                    continuation=continuation,
                )

                pages = payload.get("query", {}).get(
                    "pages",
                    [],
                )

                if not isinstance(pages, list):
                    raise TypeError("Wikimedia response pages must be a list")

                for page in pages:
                    if not isinstance(page, dict):
                        continue

                    namespace = page.get("ns")

                    if namespace == 6:
                        candidate = self._to_candidate(page)

                        if candidate is not None:
                            candidates.setdefault(
                                candidate.image_url,
                                candidate,
                            )

                    elif namespace == 14 and request.depth < maximum_category_depth:
                        self._enqueue_category(
                            page=page,
                            depth=request.depth + 1,
                            pending=pending,
                            queued_categories=queued_categories,
                            visited_categories=visited_categories,
                            maximum_categories=maximum_categories,
                        )

                    if len(candidates) >= limit:
                        break

                continuation = self._continuation_token(payload)

                if continuation is None:
                    break

        return tuple(candidates.values())

    def close(self) -> None:
        self._client.close()

    def _query_category(
        self,
        *,
        category: str,
        continuation: str | None,
    ) -> dict[str, Any]:
        parameters = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file|subcat",
            "gcmlimit": "50",
            "prop": "info|imageinfo",
            "inprop": "url",
            "iiprop": "url|mime|size|extmetadata",
        }

        if continuation is not None:
            parameters["gcmcontinue"] = continuation

        response = self._client.get(
            self._api_url,
            params=parameters,
        )
        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise TypeError("Wikimedia response must contain an object")

        return payload

    @classmethod
    def _enqueue_category(
        cls,
        *,
        page: dict[str, Any],
        depth: int,
        pending: deque[WikimediaCategoryRequest],
        queued_categories: set[str],
        visited_categories: set[str],
        maximum_categories: int,
    ) -> None:
        title = page.get("title")

        if not isinstance(title, str):
            return

        normalized_title = cls._normalize_category(title)
        category_key = normalized_title.casefold()

        if category_key in queued_categories:
            return

        category_count = len(visited_categories) + len(pending)

        if category_count >= maximum_categories:
            return

        queued_categories.add(category_key)
        pending.append(
            WikimediaCategoryRequest(
                title=normalized_title,
                depth=depth,
            )
        )

    @staticmethod
    def _continuation_token(
        payload: dict[str, Any],
    ) -> str | None:
        continuation = payload.get("continue")

        if not isinstance(continuation, dict):
            return None

        token = continuation.get("gcmcontinue")

        return token if isinstance(token, str) else None

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
            title=(cls._metadata_text(metadata, "ObjectName") or cls._page_title(page)),
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

        return title.removeprefix("File:")

    @staticmethod
    def _normalize_category(category: str) -> str:
        normalized = category.strip()

        if normalized.casefold().startswith("category:"):
            return f"Category:{normalized.split(':', maxsplit=1)[1]}"

        return f"Category:{normalized}"
