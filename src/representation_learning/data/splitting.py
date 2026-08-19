"""Deterministic training, validation, and test dataset splitting."""

from dataclasses import dataclass
from hashlib import sha256

from representation_learning.retraining.dataset_snapshot import (
    DatasetManifestItem,
)


@dataclass(frozen=True, slots=True)
class DatasetPartitions:
    train: tuple[DatasetManifestItem, ...]
    validation: tuple[DatasetManifestItem, ...]
    test: tuple[DatasetManifestItem, ...]


class DeterministicDatasetSplitter:
    def __init__(
        self,
        *,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        seed: str,
    ) -> None:
        ratios = (train_ratio, validation_ratio, test_ratio)

        if any(ratio < 0 or ratio > 1 for ratio in ratios):
            raise ValueError("Dataset split ratios must be between 0 and 1")

        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Dataset split ratios must add up to 1")

        if not seed:
            raise ValueError("Dataset split seed cannot be empty")

        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio
        self._seed = seed

    def split(
        self,
        items: tuple[DatasetManifestItem, ...],
    ) -> DatasetPartitions:
        train: list[DatasetManifestItem] = []
        validation: list[DatasetManifestItem] = []
        test: list[DatasetManifestItem] = []

        validation_boundary = self._train_ratio + self._validation_ratio

        for item in items:
            value = self._stable_value(item.image_id)

            if value < self._train_ratio:
                train.append(item)
            elif value < validation_boundary:
                validation.append(item)
            else:
                test.append(item)

        return DatasetPartitions(
            train=tuple(train),
            validation=tuple(validation),
            test=tuple(test),
        )

    def _stable_value(self, image_id: str) -> float:
        digest = sha256(
            f"{self._seed}:{image_id}".encode(),
        ).digest()

        integer = int.from_bytes(digest[:8], byteorder="big")

        return integer / 2**64
