"""Safe downloading of discovered web images."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic, sleep
from typing import ClassVar
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx

from representation_learning.domain.entities import ImageSource
from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
    normalize_web_url,
)
from representation_learning.storage.image_store import (
    ImageStore,
    StorageArea,
)


@dataclass(frozen=True, slots=True)
class DownloadedImage:
    content: bytes
    content_type: str
    extension: str
    final_url: str

    @property
    def size_bytes(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class PublishedRawImage:
    image_id: str
    storage_uri: str
    source_url: str
    size_bytes: int


class RawImagePublisher:
    def __init__(
        self,
        *,
        image_store: ImageStore,
    ) -> None:
        self._image_store = image_store

    def publish(
        self,
        *,
        candidate: ScrapedImageCandidate,
        downloaded: DownloadedImage,
    ) -> PublishedRawImage:
        image_id = str(uuid4())

        metadata = {
            "source": ImageSource.WEB_SCRAPER.value,
            "source_page_url": candidate.source_page_url,
            "image_url": downloaded.final_url,
        }

        if candidate.license_name is not None:
            metadata["license_name"] = candidate.license_name

        if candidate.creator is not None:
            metadata["creator"] = candidate.creator

        if candidate.title is not None:
            metadata["title"] = candidate.title

        storage_uri = self._image_store.save(
            image_id=image_id,
            content=downloaded.content,
            area=StorageArea.RAW,
            extension=downloaded.extension,
            metadata=metadata,
        )

        return PublishedRawImage(
            image_id=image_id,
            storage_uri=storage_uri,
            source_url=downloaded.final_url,
            size_bytes=downloaded.size_bytes,
        )


class ImageDownloader:
    _SUPPORTED_TYPES: ClassVar[dict[str, str]] = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }

    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str],
        maximum_response_bytes: int = 10_000_000,
        timeout_seconds: float = 20.0,
        maximum_redirects: int = 5,
        maximum_attempts: int = 3,
        minimum_request_interval_seconds: float = 1.0,
        maximum_retry_delay_seconds: float = 60.0,
        user_agent: str = "RepresentationLearningCrawler/0.1",
        sleep_function: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("At least one image host must be allowed")

        if maximum_response_bytes <= 0:
            raise ValueError("maximum_response_bytes must be positive")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if maximum_redirects < 0:
            raise ValueError("maximum_redirects cannot be negative")

        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")

        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds cannot be negative")

        if maximum_retry_delay_seconds <= 0:
            raise ValueError("maximum_retry_delay_seconds must be positive")

        self._allowed_hosts = frozenset(host.casefold() for host in allowed_hosts)
        self._maximum_response_bytes = maximum_response_bytes
        self._maximum_redirects = maximum_redirects
        self._maximum_attempts = maximum_attempts
        self._minimum_request_interval_seconds = minimum_request_interval_seconds
        self._maximum_retry_delay_seconds = maximum_retry_delay_seconds
        self._sleep = sleep_function
        self._clock = clock
        self._last_request_started_at: float | None = None

        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
            headers={
                "User-Agent": user_agent,
                "Accept": "image/jpeg,image/png,image/webp",
            },
        )

    def download(
        self,
        candidate: ScrapedImageCandidate,
    ) -> DownloadedImage:
        for attempt in range(self._maximum_attempts):
            try:
                return self._download_once(candidate)
            except httpx.HTTPStatusError as error:
                if (
                    error.response.status_code != 429
                    or attempt == self._maximum_attempts - 1
                ):
                    raise

                delay = self._retry_delay(
                    response=error.response,
                    attempt=attempt,
                )
                self._sleep(delay)
            except httpx.TransportError:
                if attempt == self._maximum_attempts - 1:
                    raise

                delay = min(
                    2**attempt,
                    self._maximum_retry_delay_seconds,
                )
                self._sleep(delay)

        raise RuntimeError("Unexpected image-download retry termination")

    def _download_once(
        self,
        candidate: ScrapedImageCandidate,
    ) -> DownloadedImage:
        current_url = normalize_web_url(candidate.image_url)

        for redirect_count in range(self._maximum_redirects + 1):
            self._validate_url(current_url)
            self._wait_for_request_slot()

            with self._client.stream(
                "GET",
                current_url,
            ) as response:
                if response.is_redirect:
                    if redirect_count >= self._maximum_redirects:
                        raise ValueError("Maximum redirect count exceeded")

                    location = response.headers.get("location")

                    if location is None:
                        raise ValueError("Redirect has no location")

                    current_url = normalize_web_url(urljoin(current_url, location))
                    continue

                response.raise_for_status()

                content_type = self._content_type(response)
                extension = self._SUPPORTED_TYPES.get(content_type)

                if extension is None:
                    raise ValueError(f"Unsupported image content type: {content_type}")

                self._validate_declared_size(response)

                content = bytearray()

                for chunk in response.iter_bytes():
                    content.extend(chunk)

                    if len(content) > self._maximum_response_bytes:
                        raise ValueError("Image exceeds maximum size")

                if not content:
                    raise ValueError("Downloaded image is empty")

                return DownloadedImage(
                    content=bytes(content),
                    content_type=content_type,
                    extension=extension,
                    final_url=current_url,
                )

        raise RuntimeError("Unexpected redirect-loop termination")

    def close(self) -> None:
        self._client.close()

    def _wait_for_request_slot(self) -> None:
        now = self._clock()

        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            remaining = self._minimum_request_interval_seconds - elapsed

            if remaining > 0:
                self._sleep(remaining)

        self._last_request_started_at = self._clock()

    def _retry_delay(
        self,
        *,
        response: httpx.Response,
        attempt: int,
    ) -> float:
        retry_after = response.headers.get("retry-after")

        if retry_after is not None:
            parsed_delay = self._parse_retry_after(retry_after)

            if parsed_delay is not None:
                return min(
                    parsed_delay,
                    self._maximum_retry_delay_seconds,
                )

        return min(
            2**attempt,
            self._maximum_retry_delay_seconds,
        )

    @staticmethod
    def _parse_retry_after(
        retry_after: str,
    ) -> float | None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

        try:
            retry_time = parsedate_to_datetime(retry_after)
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return None

        return max(
            (retry_time - datetime.now(UTC)).total_seconds(),
            0.0,
        )

    def _validate_url(self, url: str) -> None:
        parsed = urlsplit(url)

        if parsed.scheme != "https":
            raise ValueError("Image URL must use HTTPS")

        hostname = parsed.hostname

        if hostname is None:
            raise ValueError("Image URL has no hostname")

        normalized_hostname = hostname.casefold()

        allowed = any(
            normalized_hostname == allowed_host
            or normalized_hostname.endswith(f".{allowed_host}")
            for allowed_host in self._allowed_hosts
        )

        if not allowed:
            raise ValueError(f"Image host is not allowed: {hostname}")

    def _validate_declared_size(
        self,
        response: httpx.Response,
    ) -> None:
        content_length = response.headers.get("content-length")

        if content_length is None:
            return

        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length header") from error

        if declared_size > self._maximum_response_bytes:
            raise ValueError("Image exceeds maximum size")

    @staticmethod
    def _content_type(
        response: httpx.Response,
    ) -> str:
        return (
            response.headers.get("content-type", "")
            .split(";", maxsplit=1)[0]
            .strip()
            .casefold()
        )
