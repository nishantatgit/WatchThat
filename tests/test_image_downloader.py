import httpx
import pytest

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)
from representation_learning.scraper.image_downloader import (
    ImageDownloader,
)


def test_retries_after_rate_limit() -> None:
    request_count = 0
    sleep_delays: list[float] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        if request_count == 1:
            return httpx.Response(
                status_code=429,
                headers={"Retry-After": "2"},
                request=request,
            )

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "image/jpeg"},
            content=b"image bytes",
            request=request,
        )

    downloader = ImageDownloader(
        allowed_hosts=frozenset({"upload.wikimedia.org"}),
        maximum_attempts=3,
        minimum_request_interval_seconds=0,
        sleep_function=sleep_delays.append,
        transport=httpx.MockTransport(handle_request),
    )

    candidate = ScrapedImageCandidate(
        image_url=("https://upload.wikimedia.org/bird.jpg"),
        source_page_url=("https://commons.wikimedia.org/wiki/File:Bird.jpg"),
        license_name="CC BY-SA 4.0",
    )

    try:
        downloaded = downloader.download(candidate)
    finally:
        downloader.close()

    assert request_count == 2
    assert sleep_delays == [2.0]
    assert downloaded.content == b"image bytes"
    assert downloaded.content_type == "image/jpeg"
    assert downloaded.extension == "jpg"

def test_spaces_requests_using_minimum_interval() -> None:
    current_time = 0.0
    sleep_delays: list[float] = []

    def clock() -> float:
        return current_time

    def fake_sleep(delay: float) -> None:
        nonlocal current_time
        sleep_delays.append(delay)
        current_time += delay

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "image/jpeg"},
            content=b"image bytes",
            request=request,
        )

    downloader = ImageDownloader(
        allowed_hosts=frozenset(
            {"upload.wikimedia.org"}
        ),
        minimum_request_interval_seconds=1,
        sleep_function=fake_sleep,
        clock=clock,
        transport=httpx.MockTransport(handle_request),
    )

    candidate = ScrapedImageCandidate(
        image_url=(
            "https://upload.wikimedia.org/bird.jpg"
        ),
        source_page_url=(
            "https://commons.wikimedia.org/wiki/File:Bird.jpg"
        ),
        license_name="CC BY-SA 4.0",
    )

    try:
        downloader.download(candidate)
        downloader.download(candidate)
    finally:
        downloader.close()

    assert sleep_delays == [1.0]


def test_stops_after_maximum_attempts() -> None:
    request_count = 0
    sleep_delays: list[float] = []

    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=429,
            request=request,
        )

    downloader = ImageDownloader(
        allowed_hosts=frozenset(
            {"upload.wikimedia.org"}
        ),
        maximum_attempts=3,
        minimum_request_interval_seconds=0,
        sleep_function=sleep_delays.append,
        transport=httpx.MockTransport(handle_request),
    )

    candidate = ScrapedImageCandidate(
        image_url=(
            "https://upload.wikimedia.org/bird.jpg"
        ),
        source_page_url=(
            "https://commons.wikimedia.org/wiki/File:Bird.jpg"
        ),
        license_name="CC BY-SA 4.0",
    )

    try:
        with pytest.raises(httpx.HTTPStatusError):
            downloader.download(candidate)
    finally:
        downloader.close()

    assert request_count == 3
    assert sleep_delays == [1, 2]
