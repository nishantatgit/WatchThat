"""Exact hash and perceptual-hash image deduplication."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol


def calculate_checksum(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ValueError("Cannot calculate a checksum for empty content")

    return sha256(image_bytes).hexdigest()


class ChecksumRegistry(Protocol):
    def contains(self, checksum: str) -> bool: ...

    def add(self, checksum: str, image_id: str) -> None: ...


class InMemoryChecksumRegistry:
    def __init__(self) -> None:
        self._images_by_checksum: dict[str, str] = {}

    def contains(self, checksum: str) -> bool:
        return checksum in self._images_by_checksum

    def add(self, checksum: str, image_id: str) -> None:
        self._images_by_checksum[checksum] = image_id

    def get_image_id(self, checksum: str) -> str | None:
        return self._images_by_checksum.get(checksum)


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    checksum: str
    is_duplicate: bool
    existing_image_id: str | None = None


class ImageDeduplicator:
    def __init__(self, registry: ChecksumRegistry) -> None:
        self._registry = registry

    def check(self, image_bytes: bytes) -> DeduplicationResult:
        checksum = calculate_checksum(image_bytes)

        if self._registry.contains(checksum):
            existing_image_id = None

            get_image_id = getattr(self._registry, "get_image_id", None)
            if get_image_id is not None:
                existing_image_id = get_image_id(checksum)

            return DeduplicationResult(
                checksum=checksum,
                is_duplicate=True,
                existing_image_id=existing_image_id,
            )

        return DeduplicationResult(
            checksum=checksum,
            is_duplicate=False,
        )

    def register(self, checksum: str, image_id: str) -> None:
        self._registry.add(checksum, image_id)
