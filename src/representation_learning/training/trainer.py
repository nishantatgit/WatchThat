"""Training and validation loops."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer

from representation_learning.losses.contrastive_loss import (
    ContrastiveLoss,
)
from representation_learning.models.contrastive_model import (
    ContrastiveModel,
)


@dataclass(frozen=True, slots=True)
class TrainEpochResult:
    average_loss: float
    batch_count: int
    image_count: int


@dataclass(frozen=True, slots=True)
class EvaluationEpochResult:
    average_loss: float
    batch_count: int
    image_count: int


class ContrastiveTrainer:
    def __init__(
        self,
        *,
        model: ContrastiveModel,
        loss_function: ContrastiveLoss,
        optimizer: Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._loss_function = loss_function.to(device)
        self._optimizer = optimizer
        self._device = device

    @property
    def model(self) -> ContrastiveModel:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def train_epoch(
        self,
        data_loader: Any,
    ) -> TrainEpochResult:
        self._model.train()

        total_loss = 0.0
        total_images = 0
        batch_count = 0

        for first_views, second_views, _image_ids in data_loader:
            batch_size = first_views.shape[0]

            if batch_size < 2:
                raise ValueError(
                    "Every contrastive training batch must contain "
                    "at least two images. Use drop_last=True."
                )

            first_views = first_views.to(
                self._device,
                non_blocking=True,
            )
            second_views = second_views.to(
                self._device,
                non_blocking=True,
            )

            self._optimizer.zero_grad(set_to_none=True)

            output = self._model(
                first_views,
                second_views,
            )

            loss = self._loss_function(
                output.first_projections,
                output.second_projections,
            )

            loss.backward()
            self._optimizer.step()

            total_loss += loss.detach().item() * batch_size
            total_images += batch_size
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Training data loader produced no batches")

        return TrainEpochResult(
            average_loss=total_loss / total_images,
            batch_count=batch_count,
            image_count=total_images,
        )

    @torch.no_grad()
    def evaluate_epoch(
        self,
        data_loader: Any,
    ) -> EvaluationEpochResult:
        self._model.eval()

        total_loss = 0.0
        total_images = 0
        batch_count = 0

        for first_views, second_views, _image_ids in data_loader:
            batch_size = first_views.shape[0]

            if batch_size < 2:
                raise ValueError(
                    "Every contrastive evaluation batch must contain "
                    "at least two images. Use drop_last=True."
                )

            first_views = first_views.to(
                self._device,
                non_blocking=True,
            )
            second_views = second_views.to(
                self._device,
                non_blocking=True,
            )

            output = self._model(
                first_views,
                second_views,
            )

            loss = self._loss_function(
                output.first_projections,
                output.second_projections,
            )

            total_loss += loss.item() * batch_size
            total_images += batch_size
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Evaluation data loader produced no batches")

        return EvaluationEpochResult(
            average_loss=total_loss / total_images,
            batch_count=batch_count,
            image_count=total_images,
        )
