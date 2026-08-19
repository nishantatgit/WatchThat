"""Custom CNN and residual building blocks using basic PyTorch layers."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if stride not in {1, 2}:
            raise ValueError("stride must be either 1 or 2")

        self.main_path = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        )

        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.main_path(inputs)

        return self.activation(features + residual)
