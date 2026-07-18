from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


MKE_FEATURE_ORDER = ("onehot", "ncp", "eiip", "enac")
MKE_FEATURE_CHANNELS = {"onehot": 4, "ncp": 3, "eiip": 1, "enac": 4}


class MKEResidualBlock(nn.Module):
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
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.activation(output + identity)


class ECA1D(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, bias: int = 1):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got: {channels}")
        kernel_estimate = int(abs((math.log2(channels) + bias) / gamma))
        kernel_size = kernel_estimate if kernel_estimate % 2 else kernel_estimate + 1
        kernel_size = max(1, kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def attention(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.avg_pool(inputs)
        return self.sigmoid(self.conv(pooled.transpose(-1, -2)).transpose(-1, -2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.attention(inputs)


class FullMKEECA(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernels: Sequence[int] = (3, 5, 7),
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got: {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got: {reduction}")
        kernels = tuple(int(kernel) for kernel in spatial_kernels)
        if kernels != (3, 5, 7):
            raise ValueError(f"Full MKE-ECA spatial kernels must be (3, 5, 7), got: {kernels}")
        hidden_channels = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )
        self.spatial_convs = nn.ModuleList(
            nn.Conv1d(2, 1, kernel_size=kernel, padding=kernel // 2)
            for kernel in kernels
        )
        self.spatial_fusion = nn.Conv1d(len(kernels), 1, kernel_size=1)
        self.spatial_sigmoid = nn.Sigmoid()

    def channel_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=2)
        return self.channel_mlp(pooled).unsqueeze(-1)

    def spatial_attention(self, channel_scaled: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (
                channel_scaled.mean(dim=1, keepdim=True),
                channel_scaled.max(dim=1, keepdim=True).values,
            ),
            dim=1,
        )
        multi_scale = torch.cat([conv(descriptors) for conv in self.spatial_convs], dim=1)
        return self.spatial_sigmoid(self.spatial_fusion(multi_scale))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_scaled = inputs * self.channel_attention(inputs)
        return channel_scaled * self.spatial_attention(channel_scaled)


def _make_branch(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        MKEResidualBlock(in_channels, 64),
        MKEResidualBlock(64, out_channels),
        nn.MaxPool1d(kernel_size=2, stride=2),
        ECA1D(out_channels),
    )


class FourStreamMKEEncoder(nn.Module):
    expected_channels = 12
    expected_length = 41
    merged_channels = 112

    def __init__(
        self,
        feature_names: Sequence[str] = MKE_FEATURE_ORDER,
        use_full_mke_eca: bool = False,
    ):
        super().__init__()
        parsed_names = tuple(str(name).lower() for name in feature_names)
        if parsed_names != MKE_FEATURE_ORDER:
            raise ValueError(
                "MKE feature order must be "
                f"{MKE_FEATURE_ORDER}, got: {parsed_names}"
            )
        self.feature_names = parsed_names
        self.use_full_mke_eca = bool(use_full_mke_eca)
        self.onehot_branch = _make_branch(4, 32)
        self.ncp_branch = _make_branch(3, 32)
        self.eiip_branch = _make_branch(1, 16)
        self.enac_branch = _make_branch(4, 32)

        self.dropout1 = nn.Dropout(0.3)
        self.merged_block1 = MKEResidualBlock(self.merged_channels, 32)
        self.full_mke_eca = FullMKEECA(channels=32) if self.use_full_mke_eca else None
        self.merged_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        self.merged_block2 = MKEResidualBlock(32, 16)
        self.merged_block3 = MKEResidualBlock(16, 16)
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(16 * (self.expected_length // 4), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.85),
        )

    def split_features(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (self.expected_channels, self.expected_length):
            raise ValueError(
                "MKE encoder expected handcrafted input shaped "
                f"(batch,{self.expected_channels},{self.expected_length}), got: {tuple(inputs.shape)}"
            )
        onehot = inputs[:, 0:4, :]
        ncp = inputs[:, 4:7, :]
        eiip = inputs[:, 7:8, :]
        enac = inputs[:, 8:12, :]
        return onehot, ncp, eiip, enac

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        onehot, ncp, eiip, enac = self.split_features(inputs)
        branches = (
            self.onehot_branch(onehot),
            self.ncp_branch(ncp),
            self.eiip_branch(eiip),
            self.enac_branch(enac),
        )
        merged = self.dropout1(torch.cat(branches, dim=1))
        merged = self.merged_block1(merged)
        if self.full_mke_eca is not None:
            merged = self.full_mke_eca(merged)
        merged = self.dropout2(self.merged_pool(merged))
        merged = self.merged_block3(self.merged_block2(merged))
        return self.output(self.flatten(merged))


def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class MKEFusionHead(nn.Module):
    supported_policies = ("native", "proj256")

    def __init__(
        self,
        birna_input_dim: int,
        hand_input_dim: int = 64,
        dimension_policy: str = "native",
        native_hand_dim: int = 128,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        policy = str(dimension_policy).lower()
        if policy not in self.supported_policies:
            raise ValueError(
                f"MKE fusion dimension policy must be one of {self.supported_policies}, got: {policy}"
            )
        self.dimension_policy = policy
        if policy == "native":
            self.birna_projection = nn.Identity()
            self.hand_projection = _projection(hand_input_dim, native_hand_dim, dropout)
            classifier_input_dim = birna_input_dim + native_hand_dim
        else:
            self.birna_projection = _projection(birna_input_dim, projection_dim, dropout)
            self.hand_projection = _projection(hand_input_dim, projection_dim, dropout)
            classifier_input_dim = projection_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, birna_features: torch.Tensor, hand_features: torch.Tensor) -> torch.Tensor:
        birna_projected = self.birna_projection(birna_features)
        hand_projected = self.hand_projection(hand_features)
        return self.classifier(torch.cat((birna_projected, hand_projected), dim=1))


class MKEFeatureFusionClassifier(nn.Module):
    def __init__(
        self,
        birna_input_dim: int,
        feature_names: Sequence[str] = MKE_FEATURE_ORDER,
        dimension_policy: str = "native",
        use_full_mke_eca: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.handcrafted_encoder = FourStreamMKEEncoder(
            feature_names=feature_names,
            use_full_mke_eca=use_full_mke_eca,
        )
        self.fusion_head = MKEFusionHead(
            birna_input_dim=birna_input_dim,
            hand_input_dim=64,
            dimension_policy=dimension_policy,
            dropout=dropout,
        )

    def forward(
        self,
        birna_features: torch.Tensor,
        handcrafted_features: torch.Tensor,
    ) -> torch.Tensor:
        hand_features = self.handcrafted_encoder(handcrafted_features)
        return self.fusion_head(birna_features, hand_features)
