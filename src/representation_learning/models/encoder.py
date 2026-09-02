"""Randomly initialized CNN image encoder.
No pretrained weights or imported model architecture will be used.
"""

import torch
from torch import nn

from representation_learning.models.blocks import ResidualBlock


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feature_dimension = 512

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.stage_1 = self._make_stage(
            input_channels=64,
            output_channels=64,
            block_count=2,
            first_stride=1,
        )
        self.stage_2 = self._make_stage(
            input_channels=64,
            output_channels=128,
            block_count=2,
            first_stride=2,
        )
        self.stage_3 = self._make_stage(
            input_channels=128,
            output_channels=256,
            block_count=2,
            first_stride=2,
        )
        self.stage_4 = self._make_stage(
            input_channels=256,
            output_channels=512,
            block_count=2,
            first_stride=2,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.stage_1(features)
        features = self.stage_2(features)
        features = self.stage_3(features)
        features = self.stage_4(features)
        features = self.pool(features)

        return torch.flatten(features, start_dim=1)

    @staticmethod
    def _make_stage(
        *,
        input_channels: int,
        output_channels: int,
        block_count: int,
        first_stride: int,
    ) -> nn.Sequential:
        if block_count <= 0:
            raise ValueError("block_count must be positive")

        blocks: list[nn.Module] = [
            ResidualBlock(
                input_channels=input_channels,
                output_channels=output_channels,
                stride=first_stride,
            )
        ]

        for _ in range(1, block_count):
            blocks.append(
                ResidualBlock(
                    input_channels=output_channels,
                    output_channels=output_channels,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
