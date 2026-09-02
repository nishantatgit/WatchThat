"""Safe downloading of discovered web images."""

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urljoin, urlsplit
from uuid import uuid4
import httpx

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
    normalize_web_url,
)

from representation_learning.domain.entities import ImageSource

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
        user_agent: str = "RepresentationLearningCrawler/0.1",
    ) -> None:
        if not allowed_hosts:
            raise ValueError("At least one image host must be allowed")

        if maximum_response_bytes <= 0:
            raise ValueError("maximum_response_bytes must be positive")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if maximum_redirects < 0:
            raise ValueError("maximum_redirects cannot be negative")

        self._allowed_hosts = frozenset(host.casefold() for host in allowed_hosts)
        self._maximum_response_bytes = maximum_response_bytes
        self._maximum_redirects = maximum_redirects

        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={
                "User-Agent": user_agent,
                "Accept": "image/jpeg,image/png,image/webp",
            },
        )

    def download(
        self,
        candidate: ScrapedImageCandidate,
    ) -> DownloadedImage:
        current_url = normalize_web_url(candidate.image_url)

        for redirect_count in range(self._maximum_redirects + 1):
            self._validate_url(current_url)

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

