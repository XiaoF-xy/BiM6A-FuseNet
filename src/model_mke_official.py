from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialResidualBlock(nn.Module):
    """Residual block used by the public MKE-ResNet checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        num_groups = max(1, out_channels // 8)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.GroupNorm(num_groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        output = self.activation(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.activation(output + identity)


class OfficialECABlock(nn.Module):
    def __init__(self, channel: int, gamma: int = 2, b: int = 1):
        super().__init__()
        kernel_estimate = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_estimate if kernel_estimate % 2 else kernel_estimate + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(inputs)
        weights = self.conv(weights.transpose(-1, -2)).transpose(-1, -2)
        weights = self.sigmoid(weights)
        return inputs * weights.expand_as(inputs)


class OfficialMKEClassifier(nn.Module):
    """13-channel handcrafted classifier matching the public MKE source architecture."""

    expected_channels = 13

    def __init__(self, sequence_length: int = 41):
        super().__init__()
        self.sequence_length = int(sequence_length)

        self.res_block1_1 = OfficialResidualBlock(4, 64)
        self.res_block1_2 = OfficialResidualBlock(64, 32)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.res_block2_1 = OfficialResidualBlock(4, 64)
        self.res_block2_2 = OfficialResidualBlock(64, 32)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.res_block3_1 = OfficialResidualBlock(1, 64)
        self.res_block3_2 = OfficialResidualBlock(64, 16)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.res_block4_1 = OfficialResidualBlock(4, 64)
        self.res_block4_2 = OfficialResidualBlock(64, 32)
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.eca_block1 = OfficialECABlock(32)
        self.eca_block2 = OfficialECABlock(32)
        self.eca_block3 = OfficialECABlock(16)
        self.eca_block4 = OfficialECABlock(32)

        self.dropout1 = nn.Dropout(0.3)
        self.res_block_merged_1 = OfficialResidualBlock(112, 32)
        self.max_pool_merged_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        self.res_block_merged_2 = OfficialResidualBlock(32, 16)
        self.res_block_merged_3 = OfficialResidualBlock(16, 16)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * (self.sequence_length // 4), 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.85)
        self.fc_final_1 = nn.Linear(64, 32)
        self.bn_final_1 = nn.BatchNorm1d(32)
        self.fc_final_2 = nn.Linear(32, 2)

    def split_features(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_shape = (self.expected_channels, self.sequence_length)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                "Official MKE expected handcrafted input shaped "
                f"(batch,{self.expected_channels},{self.sequence_length}), got: {tuple(inputs.shape)}"
            )
        return (
            inputs[:, 0:4, :],
            inputs[:, 4:8, :],
            inputs[:, 8:9, :],
            inputs[:, 9:13, :],
        )

    def forward(self, handcrafted_features: torch.Tensor) -> torch.Tensor:
        onehot, chemical4, eiip, enac = self.split_features(handcrafted_features)

        onehot = self.max_pool1(self.res_block1_2(self.res_block1_1(onehot)))
        chemical4 = self.max_pool2(self.res_block2_2(self.res_block2_1(chemical4)))
        eiip = self.max_pool3(self.res_block3_2(self.res_block3_1(eiip)))
        enac = self.max_pool4(self.res_block4_2(self.res_block4_1(enac)))

        merged = torch.cat(
            (
                self.eca_block1(onehot),
                self.eca_block2(chemical4),
                self.eca_block3(eiip),
                self.eca_block4(enac),
            ),
            dim=1,
        )
        merged = self.dropout1(merged)
        merged = self.res_block_merged_1(merged)
        merged = self.dropout2(self.max_pool_merged_1(merged))
        merged = self.res_block_merged_3(self.res_block_merged_2(merged))

        features = F.relu(self.bn_fc1(self.fc1(self.flatten(merged))))
        features = self.dropout3(features)
        return self.fc_final_2(F.relu(self.bn_final_1(self.fc_final_1(features))))
