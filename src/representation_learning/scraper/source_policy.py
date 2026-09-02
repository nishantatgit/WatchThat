"""Source and licensing policy for scraped training images."""

from dataclasses import dataclass
from urllib.parse import urlsplit

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)


@dataclass(frozen=True, slots=True)
class SourcePolicyDecision:
    allowed: bool
    reason: str | None = None


class ScrapingSourcePolicy:
    def __init__(
        self,
        *,
        allowed_source_hosts: frozenset[str],
        allowed_licenses: frozenset[str],
        require_license: bool = True,
    ) -> None:
        if not allowed_source_hosts:
            raise ValueError(
                "At least one source host must be allowed"
            )

        if require_license and not allowed_licenses:
            raise ValueError(
                "At least one licence must be allowed"
            )

        self._allowed_source_hosts = frozenset(
            host.casefold()
            for host in allowed_source_hosts
        )
        self._allowed_licenses = frozenset(
            self._normalize_license(license_name)
            for license_name in allowed_licenses
        )
        self._require_license = require_license

    def evaluate(
        self,
        candidate: ScrapedImageCandidate,
    ) -> SourcePolicyDecision:
        source_decision = self._evaluate_source_url(
            candidate.source_page_url
        )

        if not source_decision.allowed:
            return source_decision

        if candidate.license_name is None:
            if self._require_license:
                return SourcePolicyDecision(
                    allowed=False,
                    reason="Licence information is missing",
                )

            return SourcePolicyDecision(allowed=True)

        normalized_license = self._normalize_license(
            candidate.license_name
        )

        if normalized_license not in self._allowed_licenses:
            return SourcePolicyDecision(
                allowed=False,
                reason=(
                    "Licence is not allowed: "
                    f"{candidate.license_name}"
                ),
            )

        return SourcePolicyDecision(allowed=True)

    def _evaluate_source_url(
        self,
        source_page_url: str,
    ) -> SourcePolicyDecision:
        parsed = urlsplit(source_page_url)

        if parsed.scheme.casefold() != "https":
            return SourcePolicyDecision(
                allowed=False,
                reason="Source page must use HTTPS",
            )

        if parsed.hostname is None:
            return SourcePolicyDecision(
                allowed=False,
                reason="Source page has no hostname",
            )

        hostname = parsed.hostname.casefold()

        allowed = any(
            hostname == allowed_host
            or hostname.endswith(f".{allowed_host}")
            for allowed_host in self._allowed_source_hosts
        )

        if not allowed:
            return SourcePolicyDecision(
                allowed=False,
                reason=f"Source host is not allowed: {hostname}",
            )

        return SourcePolicyDecision(allowed=True)

    @staticmethod
    def _normalize_license(license_name: str) -> str:
        return " ".join(
            license_name.split()
        ).casefold()