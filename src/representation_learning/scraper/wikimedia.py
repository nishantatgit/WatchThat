"""Wikimedia Commons image discovery through the MediaWiki API."""

import time as time_module
from collections import deque
from collections.abc import Callable
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


@dataclass(frozen=True, slots=True)
class WikimediaDiscoveryProgress:
    api_pages_processed: int
    candidates_found: int
    categories_visited: int
    current_category: str


class WikimediaCommonsSource:
    def __init__(
        self,
        *,
        api_url: str = "https://commons.wikimedia.org/w/api.php",
        timeout_seconds: float = 20.0,
        user_agent: str = "RepresentationLearningCrawler/0.1",
        maximum_attempts: int = 8,
        minimum_request_interval_seconds: float = 1.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")

        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")

        self._api_url = api_url
        self._maximum_attempts = maximum_attempts
        self._minimum_request_interval_seconds = minimum_request_interval_seconds
        self._last_request_started_at: float | None = None

        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
        )

    def discover(
        self,
        *,
        category: str,
        limit: int,
        maximum_category_depth: int = 0,
        maximum_categories: int = 1,
        should_include: (Callable[[ScrapedImageCandidate], bool] | None) = None,
        progress_callback: (Callable[[WikimediaDiscoveryProgress], None] | None) = None,
        progress_interval_pages: int = 10,
    ) -> tuple[ScrapedImageCandidate, ...]:
        if not category.strip():
            raise ValueError("category cannot be empty")

        if limit <= 0:
            raise ValueError("limit must be positive")

        if maximum_category_depth < 0:
            raise ValueError("maximum_category_depth cannot be negative")

        if maximum_categories <= 0:
            raise ValueError("maximum_categories must be positive")

        if progress_interval_pages <= 0:
            raise ValueError("progress_interval_pages must be positive")

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
        api_pages_processed = 0

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

                api_pages_processed += 1

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
                        candidate = self._to_candidate(
                            page,
                            source_category=request.title,
                        )

                        if candidate is None:
                            continue

                        if should_include is not None and not should_include(candidate):
                            continue

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

                if (
                    progress_callback is not None
                    and api_pages_processed % progress_interval_pages == 0
                ):
                    progress_callback(
                        WikimediaDiscoveryProgress(
                            api_pages_processed=api_pages_processed,
                            candidates_found=len(candidates),
                            categories_visited=len(visited_categories),
                            current_category=request.title,
                        )
                    )
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
            "maxlag": "5",
        }

        if continuation is not None:
            parameters["gcmcontinue"] = continuation

        last_error: httpx.HTTPError | None = None

        for attempt_index in range(self._maximum_attempts):
            self._wait_for_request_slot()

            try:
                response = self._client.get(
                    self._api_url,
                    params=parameters,
                )

                if response.status_code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    response.raise_for_status()

                    payload = response.json()

                    if not isinstance(payload, dict):
                        raise TypeError("Wikimedia response must contain an object")

                    return payload

                response.raise_for_status()
            except httpx.HTTPError as error:
                last_error = error

                if attempt_index + 1 >= self._maximum_attempts:
                    raise

                delay = self._retry_delay(
                    attempt_index=attempt_index,
                    response=error.response,
                )

                time_module.sleep(delay)

        if last_error is not None:
            raise last_error

        raise RuntimeError("Wikimedia request failed without an error")

    def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is not None:
            elapsed = time_module.monotonic() - self._last_request_started_at
            remaining = self._minimum_request_interval_seconds - elapsed

            if remaining > 0:
                time_module.sleep(remaining)

        self._last_request_started_at = time_module.monotonic()

    @staticmethod
    def _retry_delay(
        *,
        attempt_index: int,
        response: httpx.Response | None,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")

            if retry_after is not None:
                try:
                    parsed_delay = float(retry_after)
                except ValueError:
                    pass
                else:
                    if parsed_delay > 0:
                        return min(parsed_delay, 900.0)

            if response.status_code == 429:
                exponential_delay = 60.0 * (2.0**attempt_index)

                return min(exponential_delay, 900.0)

        return min(2.0**attempt_index, 60.0)

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
        *,
        source_category: str,
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
            source_category=source_category,
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
