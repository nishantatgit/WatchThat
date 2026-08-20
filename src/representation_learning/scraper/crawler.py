"""Daily image-web-scraping workflow."""

"""Core types and URL frontier for controlled web crawling."""

from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class CrawlRequest:
    url: str
    depth: int
    discovered_from: str | None = None

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("Crawl depth cannot be negative")


@dataclass(frozen=True, slots=True)
class ScrapedImageCandidate:
    image_url: str
    source_page_url: str
    license_name: str | None = None
    creator: str | None = None
    title: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedPage:
    links: tuple[str, ...]
    images: tuple[ScrapedImageCandidate, ...]


@dataclass(frozen=True, slots=True)
class CrawlFailure:
    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class CrawlResult:
    pages_downloaded: int
    pages_blocked_by_robots: int
    images: tuple[ScrapedImageCandidate, ...]
    failures: tuple[CrawlFailure, ...]


def normalize_web_url(url: str) -> str:
    parsed = urlsplit(url.strip())

    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    if parsed.hostname is None:
        raise ValueError(f"URL does not contain a hostname: {url}")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Crawler URLs must not contain credentials")

    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold()

    if parsed.port is None:
        network_location = hostname
    else:
        network_location = f"{hostname}:{parsed.port}"

    path = parsed.path or "/"

    return urlunsplit(
        (
            scheme,
            network_location,
            path,
            parsed.query,
            "",  # Fragments are not sent to the web server.
        )
    )


class HtmlPageParser:
    def parse(
        self,
        *,
        page_url: str,
        html: str,
    ) -> ParsedPage:
        normalized_page_url = normalize_web_url(page_url)
        document = BeautifulSoup(html, "html.parser")

        links: dict[str, None] = {}
        images: dict[str, ScrapedImageCandidate] = {}

        for anchor in document.find_all("a"):
            href = anchor.get("href")

            if not isinstance(href, str):
                continue

            normalized_link = self._resolve_url(
                base_url=normalized_page_url,
                value=href,
            )

            if normalized_link is not None:
                links[normalized_link] = None

        for image in document.find_all("img"):
            source = image.get("src")

            if not isinstance(source, str):
                continue

            normalized_image_url = self._resolve_url(
                base_url=normalized_page_url,
                value=source,
            )

            if normalized_image_url is None:
                continue

            title = image.get("alt")

            images[normalized_image_url] = ScrapedImageCandidate(
                image_url=normalized_image_url,
                source_page_url=normalized_page_url,
                title=title if isinstance(title, str) else None,
            )

        return ParsedPage(
            links=tuple(links),
            images=tuple(images.values()),
        )

    @staticmethod
    def _resolve_url(
        *,
        base_url: str,
        value: str,
    ) -> str | None:
        resolved_url = urljoin(base_url, value)

        try:
            return normalize_web_url(resolved_url)
        except ValueError:
            return None


class HtmlPageDownloader:
    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str],
        maximum_response_bytes: int = 2_000_000,
        timeout_seconds: float = 10.0,
        maximum_redirects: int = 5,
        user_agent: str = "RepresentationLearningCrawler/0.1",
    ) -> None:
        if not allowed_hosts:
            raise ValueError("At least one host must be allowed")

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
                "Accept": "text/html,application/xhtml+xml",
            },
        )

    def allows(self, url: str) -> bool:
        try:
            normalized_url = normalize_web_url(url)
            self._validate_host(normalized_url)
        except ValueError:
            return False

        return True

    def download(self, url: str) -> str:
        current_url = normalize_web_url(url)

        for redirect_count in range(self._maximum_redirects + 1):
            self._validate_host(current_url)

            with self._client.stream(
                "GET",
                current_url,
            ) as response:
                if response.is_redirect:
                    if redirect_count >= self._maximum_redirects:
                        raise ValueError("Maximum redirect count exceeded")

                    location = response.headers.get("location")

                    if location is None:
                        raise ValueError("Redirect response has no location")

                    current_url = normalize_web_url(urljoin(current_url, location))
                    continue

                response.raise_for_status()
                self._validate_content_type(response)

                content_length = response.headers.get("content-length")

                if (
                    content_length is not None
                    and int(content_length) > self._maximum_response_bytes
                ):
                    raise ValueError("HTML response exceeds the size limit")

                content = bytearray()

                for chunk in response.iter_bytes():
                    content.extend(chunk)

                    if len(content) > self._maximum_response_bytes:
                        raise ValueError("HTML response exceeds the size limit")

                encoding = response.charset_encoding or "utf-8"

                return bytes(content).decode(
                    encoding,
                    errors="replace",
                )

        raise RuntimeError("Unexpected redirect-loop termination")

    def close(self) -> None:
        self._client.close()

    def _validate_host(self, url: str) -> None:
        hostname = urlsplit(url).hostname

        if hostname is None:
            raise ValueError("URL has no hostname")

        normalized_hostname = hostname.casefold()

        allowed = any(
            normalized_hostname == allowed_host
            or normalized_hostname.endswith(f".{allowed_host}")
            for allowed_host in self._allowed_hosts
        )

        if not allowed:
            raise ValueError(f"Page host is not allowed: {hostname}")

    @staticmethod
    def _validate_content_type(
        response: httpx.Response,
    ) -> None:
        content_type = (
            response.headers.get("content-type", "")
            .split(";", maxsplit=1)[0]
            .strip()
            .casefold()
        )

        if content_type not in {
            "text/html",
            "application/xhtml+xml",
        }:
            raise ValueError(f"Response is not HTML: {content_type}")


class RobotsPolicy:
    def __init__(
        self,
        *,
        user_agent: str = "RepresentationLearningCrawler",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not user_agent:
            raise ValueError("user_agent cannot be empty")

        self._user_agent = user_agent
        self._rules_by_origin: dict[str, RobotFileParser] = {}
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/plain",
            },
        )

    def can_fetch(self, url: str) -> bool:
        normalized_url = normalize_web_url(url)
        parsed = urlsplit(normalized_url)

        origin = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                "",
                "",
                "",
            )
        )

        rules = self._rules_by_origin.get(origin)

        if rules is None:
            rules = self._download_rules(origin)
            self._rules_by_origin[origin] = rules

        return rules.can_fetch(
            self._user_agent,
            normalized_url,
        )

    def close(self) -> None:
        self._client.close()

    def _download_rules(
        self,
        origin: str,
    ) -> RobotFileParser:
        robots_url = f"{origin}/robots.txt"
        rules = RobotFileParser()
        rules.set_url(robots_url)

        try:
            response = self._client.get(robots_url)
        except httpx.HTTPError:
            return self._deny_all_rules(robots_url)

        if response.status_code == 404:
            return self._allow_all_rules(robots_url)

        if response.status_code != 200:
            return self._deny_all_rules(robots_url)

        rules.parse(response.text.splitlines())

        return rules

    @staticmethod
    def _allow_all_rules(
        robots_url: str,
    ) -> RobotFileParser:
        rules = RobotFileParser()
        rules.set_url(robots_url)
        rules.parse(
            [
                "User-agent: *",
                "Disallow:",
            ]
        )

        return rules

    @staticmethod
    def _deny_all_rules(
        robots_url: str,
    ) -> RobotFileParser:
        rules = RobotFileParser()
        rules.set_url(robots_url)
        rules.parse(
            [
                "User-agent: *",
                "Disallow: /",
            ]
        )

        return rules


class InMemoryUrlFrontier:
    """Bounded development implementation of a crawler URL frontier."""

    def __init__(
        self,
        *,
        maximum_urls: int,
        maximum_depth: int,
    ) -> None:
        if maximum_urls <= 0:
            raise ValueError("maximum_urls must be positive")

        if maximum_depth < 0:
            raise ValueError("maximum_depth cannot be negative")

        self._maximum_urls = maximum_urls
        self._maximum_depth = maximum_depth
        self._pending: deque[CrawlRequest] = deque()
        self._seen: set[str] = set()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def add(self, request: CrawlRequest) -> bool:
        if request.depth > self._maximum_depth:
            return False

        normalized_url = normalize_web_url(request.url)

        if normalized_url in self._seen:
            return False

        if len(self._seen) >= self._maximum_urls:
            return False

        normalized_request = CrawlRequest(
            url=normalized_url,
            depth=request.depth,
            discovered_from=request.discovered_from,
        )

        self._seen.add(normalized_url)
        self._pending.append(normalized_request)

        return True

    def next(self) -> CrawlRequest | None:
        if not self._pending:
            return None

        return self._pending.popleft()


class WebCrawler:
    def __init__(
        self,
        *,
        frontier: InMemoryUrlFrontier,
        downloader: HtmlPageDownloader,
        parser: HtmlPageParser,
        robots_policy: RobotsPolicy,
    ) -> None:
        self._frontier = frontier
        self._downloader = downloader
        self._parser = parser
        self._robots_policy = robots_policy

    def crawl(
        self,
        seed_urls: tuple[str, ...],
    ) -> CrawlResult:
        if not seed_urls:
            raise ValueError("At least one seed URL is required")

        for seed_url in seed_urls:
            if not self._downloader.allows(seed_url):
                raise ValueError(f"Seed URL is not allowed: {seed_url}")

            self._frontier.add(
                CrawlRequest(
                    url=seed_url,
                    depth=0,
                )
            )

        pages_downloaded = 0
        pages_blocked_by_robots = 0
        images: dict[str, ScrapedImageCandidate] = {}
        failures: list[CrawlFailure] = []

        while True:
            request = self._frontier.next()

            if request is None:
                break

            if not self._robots_policy.can_fetch(request.url):
                pages_blocked_by_robots += 1
                continue

            try:
                html = self._downloader.download(request.url)

                parsed_page = self._parser.parse(
                    page_url=request.url,
                    html=html,
                )
            except (httpx.HTTPError, ValueError) as error:
                failures.append(
                    CrawlFailure(
                        url=request.url,
                        reason=str(error),
                    )
                )
                continue

            pages_downloaded += 1

            for image in parsed_page.images:
                images.setdefault(
                    image.image_url,
                    image,
                )

            next_depth = request.depth + 1

            for link in parsed_page.links:
                if not self._downloader.allows(link):
                    continue

                self._frontier.add(
                    CrawlRequest(
                        url=link,
                        depth=next_depth,
                        discovered_from=request.url,
                    )
                )

        return CrawlResult(
            pages_downloaded=pages_downloaded,
            pages_blocked_by_robots=(pages_blocked_by_robots),
            images=tuple(images.values()),
            failures=tuple(failures),
        )

    def close(self) -> None:
        self._downloader.close()
        self._robots_policy.close()
