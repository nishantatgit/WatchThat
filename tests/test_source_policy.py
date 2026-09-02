from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)
from representation_learning.scraper.source_policy import (
    ScrapingSourcePolicy,
)


def create_policy() -> ScrapingSourcePolicy:
    return ScrapingSourcePolicy(
        allowed_source_hosts=frozenset({"commons.wikimedia.org"}),
        allowed_licenses=frozenset(
            {
                "CC0",
                "CC BY 4.0",
                "CC BY-SA 4.0",
                "CC BY-SA 3.0",
                "Public domain",
            }
        ),
    )


def test_allows_approved_source_and_license() -> None:
    candidate = ScrapedImageCandidate(
        image_url="https://upload.wikimedia.org/bird.jpg",
        source_page_url=("https://commons.wikimedia.org/wiki/File:Bird.jpg"),
        license_name="CC BY-SA 4.0",
    )

    decision = create_policy().evaluate(candidate)

    assert decision.allowed
    assert decision.reason is None


def test_rejects_missing_license() -> None:
    candidate = ScrapedImageCandidate(
        image_url="https://upload.wikimedia.org/bird.jpg",
        source_page_url=("https://commons.wikimedia.org/wiki/File:Bird.jpg"),
    )

    decision = create_policy().evaluate(candidate)

    assert not decision.allowed
    assert decision.reason == "Licence information is missing"


def test_rejects_unapproved_license() -> None:
    candidate = ScrapedImageCandidate(
        image_url="https://upload.wikimedia.org/bird.jpg",
        source_page_url=("https://commons.wikimedia.org/wiki/File:Bird.jpg"),
        license_name="All rights reserved",
    )

    decision = create_policy().evaluate(candidate)

    assert not decision.allowed
    assert decision.reason == ("Licence is not allowed: All rights reserved")


def test_rejects_unapproved_source_host() -> None:
    candidate = ScrapedImageCandidate(
        image_url="https://images.example.org/bird.jpg",
        source_page_url="https://example.org/bird",
        license_name="CC BY-SA 4.0",
    )

    decision = create_policy().evaluate(candidate)

    assert not decision.allowed
    assert decision.reason == ("Source host is not allowed: example.org")
