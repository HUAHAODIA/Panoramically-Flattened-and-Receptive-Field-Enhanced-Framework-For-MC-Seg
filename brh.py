"""Boundary Refinement Head (BRH) for 3D medical image segmentation.

BRH is the boundary refinement module of the segmentation network proposed in
the paper. It is attached to the last (full-resolution) decoder stage, i.e.
directly before the final segmentation layer, and enlarges the effective
receptive field at that resolution so that thin or low-contrast boundaries can
be recovered.

Design overview
---------------
``LKA_3D`` (large kernel attention) decomposes a large spatial kernel into a
depthwise convolution, a dilated depthwise convolution and a 1x1 pointwise
convolution. With ``kernel_size=7`` and ``dilation=3`` the dilated depthwise
convolution covers an effective receptive field of 19x19x19 while only using a
handful of parameters. The result is used as a multiplicative attention map
applied to the (cloned) input, i.e. ``x * attn(x)``.

``BRH`` wraps that operator in a Transformer-style residual block:

    x -> 1x1 conv -> GELU -> LKA_3D -> 1x1 conv -> + x

The second 1x1 convolution is zero-initialised, so the block starts as an exact
identity mapping and is then learned as a residual refinement of the decoder
features. This keeps training stable when the head is inserted into a
pre-trained or randomly initialised U-Net.

The block is 3D-only (``nn.Conv3d``).

Reference
---------
Cross-Level Semantic Integration Block (CSIB) and Boundary Refinement Head (BRH),
as described in the paper. See ``README.md`` for the nnU-Net integration guide.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BRH", "LKA_3D"]


class LKA_3D(nn.Module):
    """3D large kernel attention (depthwise + dilated depthwise + pointwise).

    Args:
        dim: Number of input (and output) channels.

    Shape:
        input / output -> ``(B, dim, D, H, W)``; spatial size is preserved.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Local depthwise convolution.
        self.conv0 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # Dilated depthwise convolution, equivalent to a 19x19x19 kernel.
        self.conv_spatial = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9,
                                      groups=dim, dilation=3)
        # Pointwise (channel mixing) convolution.
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class BRH(nn.Module):
    """Boundary Refinement Head.

    Args:
        d_model: Number of feature channels of the decoder stage the head is
            applied to. In nnU-Net this is ``encoder.output_channels[0]`` when
            the head is attached to the last decoder stage.

    Shape:
        input / output -> ``(B, d_model, D, H, W)``; spatial size is preserved.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA_3D(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, kernel_size=1)

        # Zero-initialise the output projection so that BRH starts as an
        # identity mapping (residual refinement).
        nn.init.constant_(self.proj_2.weight, 0)
        nn.init.constant_(self.proj_2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x
