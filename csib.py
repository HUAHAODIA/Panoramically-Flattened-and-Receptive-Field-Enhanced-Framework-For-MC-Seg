"""Cross-Level Semantic Integration Block (CSIB) for 3D medical image segmentation.

CSIB is the bottleneck module of the segmentation network proposed in the paper.
It is inserted between the deepest encoder stage and the decoder: instead of
passing only the deepest (lowest-resolution) feature map to the decoder, CSIB
collects the skip features produced by *every* encoder stage, aligns them to the
bottleneck resolution, and fuses them into a single semantically enriched
bottleneck representation.

Design overview
---------------
1. ``AMFF3D_Dynamic`` (adaptive multi-scale feature fusion)
   Each encoder skip is projected to a common channel width by a 1x1 convolution
   and resampled to the bottleneck resolution with trilinear interpolation. A
   lightweight gate predicts one sigmoid weight per branch from the globally
   pooled branch descriptors, and every branch is modulated in a residual manner
   (``f * (1 + w)``) before concatenation. The network therefore learns, per
   input volume, how much each level contributes to the fused representation.
2. ``ParallelDualAttention3D``
   The fused multi-level tensor is refined by channel attention and spatial
   attention running in parallel. A learned gate blends the two attention
   outputs with the identity mapping.
3. ``CSIB``
   Wraps (1) and (2) and projects the result back to the bottleneck channel
   width, so that the block is a drop-in replacement for the deepest encoder
   feature map.

The block is 3D-only (``nn.Conv3d`` + trilinear interpolation).

Reference
---------
Cross-Level Semantic Integration Block (CSIB) and Boundary Refinement Head (BRH),
as described in the paper. See ``README.md`` for the nnU-Net integration guide.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CSIB",
    "ChannelAttention3D",
    "SpatialAttention3D",
    "AMFF3D_Dynamic",
    "ParallelDualAttention3D",
]


class ChannelAttention3D(nn.Module):
    """Squeeze-and-excitation style channel attention for 3D feature maps.

    The 3D feature map is globally average pooled to a per-channel descriptor,
    passed through a small bottleneck MLP and used to rescale the input.
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention3D(nn.Module):
    """Spatial attention computed from average- and max-pooled feature maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return x * self.sigmoid(x_out)


class AMFF3D_Dynamic(nn.Module):
    """Adaptive multi-scale feature fusion (AMFF) for 3D encoder skips.

    Args:
        input_channels_list: Number of channels of every input feature map,
            ordered from the highest-resolution stage to the bottleneck.
        target_channels_per_branch: Channel width that every branch is projected
            to before fusion.
    """

    def __init__(self, input_channels_list: Sequence[int], target_channels_per_branch: int):
        super().__init__()
        self.num_branches = len(input_channels_list)
        self.target_c = target_channels_per_branch

        # Per-branch 1x1 projection that unifies the channel dimension.
        self.adapters = nn.ModuleList([
            nn.Conv3d(in_c, target_channels_per_branch, kernel_size=1, bias=False)
            for in_c in input_channels_list
        ])

        # Per-volume gate that predicts one weight per branch.
        self.gate = nn.Sequential(
            nn.Linear(target_channels_per_branch * self.num_branches,
                      self.num_branches, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # The deepest feature map defines the common spatial resolution.
        target_size = inputs[-1].shape[2:]

        # Step 1: spatial alignment + channel unification.
        adapted = []
        for i, feat in enumerate(inputs):
            if list(feat.shape[2:]) != list(target_size):
                # NOTE: F.interpolate is used instead of F.adaptive_max_pool3d
                # because it is fully compatible with torch.compile / TorchDynamo.
                feat = F.interpolate(feat, size=target_size,
                                     mode="trilinear", align_corners=False)
            feat = self.adapters[i](feat)
            adapted.append(feat)

        # Step 2: dynamic branch weights from globally pooled descriptors.
        # NOTE: tensor.mean is used instead of F.adaptive_avg_pool3d; both are
        # equivalent here and the former is friendlier to TorchDynamo.
        pooled = [f.mean(dim=[2, 3, 4]) for f in adapted]   # [B, C] per branch
        gate_input = torch.cat(pooled, dim=1)               # [B, C * num_branches]
        weights = self.gate(gate_input)                     # [B, num_branches]

        # Step 3: residual (identity-centred) modulation of every branch.
        weighted = []
        for i, feat in enumerate(adapted):
            w = weights[:, i].view(-1, 1, 1, 1, 1)
            weighted.append(feat * (1.0 + w))

        return torch.cat(weighted, dim=1)


class ParallelDualAttention3D(nn.Module):
    """Parallel channel + spatial attention with a learned blending gate.

    The output is ``x + alpha * (ca(x) - x) + beta * (sa(x) - x)``, so the module
    degenerates to the identity mapping when both gates are zero.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels)
        self.spatial_att = SpatialAttention3D()

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, 2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca_out = self.channel_att(x)
        sa_out = self.spatial_att(x)

        gates = self.gate(x)                             # [B, 2]
        alpha = gates[:, 0].view(-1, 1, 1, 1, 1)
        beta = gates[:, 1].view(-1, 1, 1, 1, 1)

        out = x + alpha * (ca_out - x) + beta * (sa_out - x)
        return out


class CSIB(nn.Module):
    """Cross-Level Semantic Integration Block.

    Args:
        encoder_channels_list: Number of channels of each encoder stage output,
            ordered from the stem / highest resolution stage to the bottleneck.
            In nnU-Net this is ``encoder.output_channels`` (i.e. the
            ``features_per_stage`` list).
        bottleneck_channels: Number of channels of the deepest encoder stage.

    Shape:
        input  -> list of ``len(encoder_channels_list)`` tensors
        output -> ``(B, bottleneck_channels, D, H, W)``, matching the spatial
        size of the last input tensor.

    Note:
        In the released nnU-Net integration the block output is concatenated with
        the original bottleneck feature map and fused by a 1x1 convolution, see
        ``PlainConvEncoder_CSIB`` in ``README.md``.
    """

    def __init__(self, encoder_channels_list: Sequence[int], bottleneck_channels: int):
        super().__init__()
        self.num_branches = len(encoder_channels_list)
        # Every encoder level contributes an equal share of the fused width.
        self.branch_channels = max(1, bottleneck_channels // self.num_branches)

        self.amff = AMFF3D_Dynamic(encoder_channels_list, self.branch_channels)

        self.fused_channels = self.branch_channels * self.num_branches

        self.dual_att = ParallelDualAttention3D(self.fused_channels)

        self.out_conv = nn.Conv3d(self.fused_channels, bottleneck_channels, kernel_size=1)

    def forward(self, stage_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Fuse the encoder skips into a single bottleneck feature map.

        Args:
            stage_outputs: Skip features of all encoder stages, ordered from the
                highest-resolution stage to the bottleneck (the order returned by
                an nnU-Net encoder with ``return_skips=True``).
        """
        fused = self.amff(stage_outputs)
        fused = self.dual_att(fused)
        out = self.out_conv(fused)
        return out
