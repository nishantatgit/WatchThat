"""MLflow experiment and model-artifact tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from representation_learning.training.checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
)
from representation_learning.training.trainer import (
    ContrastiveTrainer,
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    maximum_epochs: int = 50
    patience: int = 5
    minimum_improvement: float = 0.001

    def __post_init__(self) -> None:
        if self.maximum_epochs <= 0:
            raise ValueError("maximum_epochs must be positive")

        if self.patience <= 0:
            raise ValueError("patience must be positive")

        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement cannot be negative")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    training_loss: float
    validation_loss: float
    improved: bool


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path
    history: tuple[EpochMetrics, ...]


class TrainingExperiment:
    def __init__(
        self,
        *,
        trainer: ContrastiveTrainer,
        checkpoint_manager: CheckpointManager,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self._trainer = trainer
        self._checkpoint_manager = checkpoint_manager
        self._config = config
        self._device = device

    def run(
        self,
        *,
        training_loader: Any,
        validation_loader: Any,
        dataset_snapshot_id: str,
    ) -> ExperimentResult:
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        checkpoint_path: Path | None = None
        history: list[EpochMetrics] = []

        for epoch in range(1, self._config.maximum_epochs + 1):
            training_result = self._trainer.train_epoch(training_loader)
            validation_result = self._trainer.evaluate_epoch(validation_loader)

            improved = (
                validation_result.average_loss
                < best_validation_loss - self._config.minimum_improvement
            )

            history.append(
                EpochMetrics(
                    epoch=epoch,
                    training_loss=training_result.average_loss,
                    validation_loss=(validation_result.average_loss),
                    improved=improved,
                )
            )

            if improved:
                best_validation_loss = validation_result.average_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                checkpoint_path = self._checkpoint_manager.save(
                    model=self._trainer.model,
                    optimizer=self._trainer.optimizer,
                    metadata=CheckpointMetadata(
                        epoch=epoch,
                        training_loss=(training_result.average_loss),
                        validation_loss=(validation_result.average_loss),
                        dataset_snapshot_id=(dataset_snapshot_id),
                    ),
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._config.patience:
                break

        if checkpoint_path is None:
            raise RuntimeError("Experiment did not produce a checkpoint")

        self._checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=self._trainer.model,
            optimizer=self._trainer.optimizer,
            device=self._device,
        )

        return ExperimentResult(
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            epochs_completed=len(history),
            stopped_early=(len(history) < self._config.maximum_epochs),
            checkpoint_path=checkpoint_path,
            history=tuple(history),
        )
