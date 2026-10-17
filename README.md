# CSIB & BRH — Reference Implementation

Reference code for the two modules introduced in our paper:

| Module | Paper name | File | Role in the network |
| --- | --- | --- | --- |
| **CSIB** | Cross-Level Semantic Integration Block | `csib.py` | Encoder bottleneck |
| **BRH** | Boundary Refinement Head | `brh.py` | Decoder, full-resolution stage |

Both modules are designed as drop-in additions to the 3D U-Net backbone of
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (`dynamic_network_architectures`).
The training pipeline itself is untouched — no loss, no data augmentation and no
trainer subclass are changed. The network is reproduced by (i) adding the two
blocks, (ii) wiring them into `PlainConvUNet` / `UNetDecoder`, and (iii) setting
the deeper-stage kernel sizes to `7x7x7` in the nnU-Net plans file (step 6).

```
                    encoder skips (one per stage, low -> high resolution)
                                     │
   input ──► encoder stages ──► ┌────────┐
             (PlainConvEncoder) │  CSIB  │──► fused bottleneck
                                └────────┘         │
                                     │             │
                                     └── concat ───┘──► 1x1 conv ──► bottleneck
                                                                       │
                                                       decoder stages (bottom-up)
                                                                       │
                                                     full-resolution decoder stage
                                                                       │
                                                                  ┌─────────┐
                                                                  │   BRH   │
                                                                  └─────────┘
                                                                       │
                                                             segmentation head
```

---

## Contents

```
.
├── csib.py      # CSIB: Cross-Level Semantic Integration Block
├── brh.py       # BRH:  Boundary Refinement Head
└── README.md    # this file
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- `dynamic_network_architectures` (the backbone package used by nnU-Net v2)

No additional third-party dependencies. **Both modules are 3D-only**
(`nn.Conv3d`, trilinear interpolation); they target the `3d_fullres` /
`3d_lowres` / `3d_cascade_fullres` configurations and are not compatible with
the 2D configurations of nnU-Net.

---

## 1. Module interfaces

### 1.1 CSIB — `csib.py`

```python
CSIB(encoder_channels_list: Sequence[int], bottleneck_channels: int)
```

- **Input**: a `list` of `torch.Tensor`s — the skip features of *all* encoder
  stages, ordered from the highest-resolution stage (stem) to the bottleneck.
  This is exactly the list an nnU-Net encoder returns with `return_skips=True`.
- **Output**: `torch.Tensor` of shape `(B, bottleneck_channels, D, H, W)`, the
  spatial size of the last (deepest) input tensor.

Internally the block is composed of three stages:

1. `AMFF3D_Dynamic` — every skip is projected to a common channel width by a
   1x1 convolution and resampled to the bottleneck resolution with trilinear
   interpolation. A gate predicts one sigmoid weight per branch from the
   globally pooled branch descriptors, and each branch is modulated residually
   (`f * (1 + w)`) before concatenation.
2. `ParallelDualAttention3D` — channel attention and spatial attention run in
   parallel on the fused tensor, blended with the identity mapping by a learned
   two-way gate: `x + α·(ca(x) − x) + β·(sa(x) − x)`.
3. `out_conv` — a 1x1 convolution projecting the fused width back to
   `bottleneck_channels`.

The per-branch channel width is `max(1, bottleneck_channels // len(encoder_channels_list))`.

### 1.2 BRH — `brh.py`

```python
BRH(d_model: int)
```

- **Input / Output**: `torch.Tensor` of shape `(B, d_model, D, H, W)`; the
  spatial size is preserved.

The head is a Transformer-style residual block whose spatial operator is
`LKA_3D` (large kernel attention: depthwise 5³ + dilated depthwise with
`kernel_size=7, dilation=3`, i.e. an effective receptive field of 19³, + 1x1
pointwise):

```
x ──► 1x1 conv ──► GELU ──► LKA_3D ──► 1x1 conv ──► + x
```

The second 1x1 convolution is **zero-initialised**, so BRH starts as an exact
identity mapping and is learned as a residual refinement of the decoder
features. This keeps optimisation stable when the head is inserted into an
otherwise standard U-Net.

---

## 2. Where the files go

Both files belong to the **`building_blocks`** folder of
`dynamic_network_architectures`:

```
dynamic_network_architectures/
├── architectures/
│   └── unet.py                        ← STEP 4: modify PlainConvUNet
└── building_blocks/
    ├── csib.py                        ← copy from this repo
    ├── brh.py                         ← copy from this repo
    ├── plain_conv_encoder_csib.py     ← STEP 2: new file (code below)
    ├── unet_decoder.py                ← STEP 3: add the BRH head
    ├── simple_conv_blocks.py          (unchanged, stock)
    ├── helper.py                      (unchanged, stock)
    └── plain_conv_encoder.py          (unchanged, stock)
```

> If `dynamic_network_architectures` is installed as a pip package, edit the
> files inside the installed package (or install it in editable mode from the
> `dynamic-network-architecture` submodule shipped with nnU-Net v2).

---

## 3. Integration into nnU-Net, step by step

Only four files are touched. Steps 1–3 add the modules and the two host
modules; step 4 wires them into `PlainConvUNet`.

### Step 1 — Copy the module files

```bash
cp csib.py brh.py <path-to>/dynamic_network_architectures/building_blocks/
```

### Step 2 — Add the CSIB-augmented encoder

Create `dynamic_network_architectures/building_blocks/plain_conv_encoder_csib.py`.
This is a copy of the stock `PlainConvEncoder` whose bottleneck output is
refined by CSIB. **The output channel count is unchanged**, so it is a drop-in
replacement for `PlainConvEncoder`.

```python
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.csib import CSIB


class PlainConvEncoder_CSIB(nn.Module):
    """Convolutional encoder whose bottleneck is refined by CSIB.

    The encoder stages are identical to those of ``PlainConvEncoder``. The output
    of the deepest stage is additionally refined by CSIB, which aggregates the
    skip features of all encoder stages. The CSIB output is concatenated with the
    original bottleneck and fused by a 1x1 convolution, so the number of output
    channels is unchanged and this class is a drop-in replacement for
    ``PlainConvEncoder``.
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],  # must be nn.Conv3d, CSIB is 3D-only
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages, "kernel_sizes entry mismatch"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage entry mismatch"
        assert len(features_per_stage) == n_stages, "features_per_stage entry mismatch"
        assert len(strides) == n_stages, "strides entry mismatch"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s])
                    )
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()

            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # Attributes queried by UNetDecoder.
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        # ------------------------------------------------------------------
        # CSIB at the bottleneck
        # ------------------------------------------------------------------
        bottleneck_channels = features_per_stage[-1]

        # Aggregates the skip features of every encoder stage.
        self.csib = CSIB(
            encoder_channels_list=features_per_stage,
            bottleneck_channels=bottleneck_channels,
        )

        # Fuses concat(original bottleneck, CSIB output) -> bottleneck channels.
        self.final_bottleneck_conv = nn.Conv3d(
            in_channels=bottleneck_channels * 2,
            out_channels=bottleneck_channels,
            kernel_size=1,
            bias=conv_bias,
        )

    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s(x)
            skips.append(x)

        # Cross-level semantic integration at the bottleneck.
        original_bottleneck = skips[-1]
        csib_feature = self.csib(skips)
        fused = torch.cat([original_bottleneck, csib_feature], dim=1)
        skips[-1] = self.final_bottleneck_conv(fused)

        if self.return_skips:
            return skips
        return skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
```

### Step 3 — Add the BRH head to the decoder

Modify `dynamic_network_architectures/building_blocks/unet_decoder.py`. BRH is
applied to the **last** decoder stage only, i.e. the full-resolution stage right
before the segmentation layer:

```diff
--- a/dynamic_network_architectures/building_blocks/unet_decoder.py
+++ b/dynamic_network_architectures/building_blocks/unet_decoder.py
@@ -9,6 +9,7 @@
 from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
 from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
 from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
+from dynamic_network_architectures.building_blocks.brh import BRH

 class UNetDecoder(nn.Module):
@@ -97,6 +98,8 @@
         self.stages = nn.ModuleList(stages)
         self.transpconvs = nn.ModuleList(transpconvs)
         self.seg_layers = nn.ModuleList(seg_layers)
+        # BRH refines the full-resolution decoder stage (boundary refinement).
+        self.brh = BRH(d_model=encoder.output_channels[0])

     def forward(self, skips):
@@ -114,6 +117,9 @@
             x = self.stages[s](x)

+            if s == len(self.stages) - 1:
+                x = self.brh(x)
+
             if self.deep_supervision:
                 seg_outputs.append(self.seg_layers[s](x))
             elif s == (len(self.stages) - 1):
```

`encoder.output_channels[0]` is the channel count of the full-resolution
decoder output (in nnU-Net's default plans `features_per_stage[0]`, e.g. 32),
which is exactly the number of channels `BRH` must accept.

### Step 4 — Modify `unet.py`

Modify `dynamic_network_architectures/architectures/unet.py`, inside
`PlainConvUNet`. Two changes only: the import, and the encoder class.

```diff
--- a/dynamic_network_architectures/architectures/unet.py
+++ b/dynamic_network_architectures/architectures/unet.py
@@ -19,7 +19,7 @@
-from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
+from dynamic_network_architectures.building_blocks.plain_conv_encoder_csib import PlainConvEncoder_CSIB
 from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
 from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
 from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
@@ -81,7 +81,7 @@
-        self.encoder = PlainConvEncoder(
+        self.encoder = PlainConvEncoder_CSIB(
             input_channels,
             n_stages,
             features_per_stage,
```

(The old `PlainConvEncoder` import may simply be kept alongside the new one if
any other class in `unet.py` still uses it — in the stock file only
`PlainConvUNet` does.)

The decoder instantiation is untouched — `UNetDecoder` already contains the BRH
head after step 3:

```python
self.decoder = UNetDecoder(
    self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first
)
```

Nothing else in `PlainConvUNet` needs to change:
`key_to_encoder = "encoder.stages"`, `key_to_stem = "encoder.stages.0"` and
`keys_to_in_proj` all remain valid, because CSIB only post-processes the
bottleneck and does not alter the stage layout.

### Step 5 — Architecture selection in the plans file

nnU-Net selects the network through
`architecture.network_class_name` in the generated `nnUNetPlans.json`, which for
the standard 3D configuration is:

```json
"network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet"
```

Because steps 1–4 modify `PlainConvUNet` **in place**, every existing plans file
and every stock trainer (`nnUNetTrainer`) automatically builds the CSIB + BRH
network. **No new trainer and no new plans file are required** — but the kernel
sizes of the deeper stages *do* have to be configured in the plans file, see
step 6.

<details>
<summary>Alternative: register a separate architecture instead of patching <code>PlainConvUNet</code></summary>

If you prefer not to touch the stock `PlainConvUNet`, copy it into a new class
(e.g. `CSIBBRHUNet`) in a new module
`dynamic_network_architectures/architectures/unet_csib_brh.py` and point the
plans file at it:

```json
"architecture": {
    "network_class_name": "dynamic_network_architectures.architectures.unet_csib_brh.CSIBBRHUNet",
    ...
}
```

The decoder must then still be patched (step 3), or subclassed from the patched
`UNetDecoder`, since BRH lives inside the decoder.

</details>

### Step 6 — Network configuration in the plans file (kernel sizes)

The `7x7x7` kernels of the deeper encoder stages are **not** hard-coded in
Python: they are part of the nnU-Net plans file and have to be set there. Edit

```
<nnUNet_preprocessed>/Dataset<ID>_<name>/nnUNetPlans.json
```

and change the kernel sizes of the `3d_fullres` configuration, i.e. the path
`configurations` → `3d_fullres` → `architecture` → `arch_kwargs` →
`kernel_sizes`:

```diff
"3d_fullres": {
    ...
    "architecture": {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
-           "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
+           "kernel_sizes": [[3, 3, 3], [3, 3, 3], [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
            ...
```

The complete `architecture` block used in our experiments:

```json
"architecture": {
    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
    "arch_kwargs": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [
            [3, 3, 3],
            [3, 3, 3],
            [7, 7, 7],
            [7, 7, 7],
            [7, 7, 7],
            [7, 7, 7]
        ],
        "strides": [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 1, 2]
        ],
        "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
        "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
        "conv_bias": true,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-05, "affine": true},
        "dropout_op": null,
        "dropout_op_kwargs": null,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": true}
    },
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
}
```

> `features_per_stage`, `strides` and `n_conv_per_stage` are **produced by
> nnU-Net's planner** and depend on the dataset (in the example above the data
> is anisotropic, hence `features_per_stage[-1] = 320` and a final `[2, 1, 2]`
> stride, which pools in-plane but not along z). Only `kernel_sizes` is set
> manually: keep the planner's values for everything else and just replace the
> four deepest kernel sizes by `[7, 7, 7]`.

What this configuration implies:

- **Encoder.** Stages 0 and 1 (highest resolution, where large dense kernels are
  the most expensive) keep `3x3x3`; stages 2–5 use `7x7x7`. Enlarging the kernel
  at low resolution is cheap and widens the receptive field exactly where the
  network reasons about coarse context.
- **Decoder.** `UNetDecoder` builds its conv blocks with
  `encoder.kernel_sizes[-(s + 1)]`, i.e. it inherits the encoder's list in
  reverse order. With the list above, the **three deepest** decoder stages use
  `7x7x7` while the **two full-resolution** decoder stages use `3x3x3`. This is
  precisely why BRH is inserted at the full-resolution stage: the dense kernels
  there stay small, and BRH supplies a 19³ effective receptive field through
  cheap dilated depthwise convolutions instead of an expensive `7x7x7` dense
  convolution on large feature maps.
- **Other configurations.** `2d` and `3d_lowres` keep the stock `3x3(x3)`
  kernels; `3d_cascade_fullres` declares `"inherits_from": "3d_fullres"` and
  therefore also uses `7x7x7`.

> ⚠️ Re-running `nnUNetv2_plan_and_preprocess` **regenerates** `nnUNetPlans.json`
> and silently reverts `kernel_sizes` to the stock `3x3x3`. Re-apply this edit
> afterwards, or keep the modified plans file under version control. A custom
> `ExperimentPlanner` can make the change persistent, but editing the generated
> JSON is what our experiments used.

---

## 4. Sanity check

The blocks can also be instantiated standalone (no nnU-Net required):

```python
import torch
from csib import CSIB
from brh import BRH

# CSIB at the bottleneck: one tensor per encoder stage, low -> high resolution.
features_per_stage = [32, 64, 128, 256, 320]
skips = [
    torch.randn(1, c, 64 // 2 ** i, 64 // 2 ** i, 64 // 2 ** i)
    for i, c in enumerate(features_per_stage)
]
csib = CSIB(encoder_channels_list=features_per_stage, bottleneck_channels=features_per_stage[-1])
print(csib(skips).shape)   # torch.Size([1, 320, 4, 4, 4])

# BRH at the full-resolution decoder stage.
brh = BRH(d_model=features_per_stage[0])
x = torch.randn(1, 32, 64, 64, 64)
print(brh(x).shape)        # torch.Size([1, 32, 64, 64, 64])
```

Expected properties:

- `csib(skips)` returns the spatial size of the deepest skip and
  `bottleneck_channels` channels.
- `brh(x)` preserves shape and is an **identity mapping right after
  initialisation** (the output projection is zero-initialised).

---

## 5. Notes and caveats

- **3D only.** Both blocks use `nn.Conv3d`; CSIB additionally uses
  `mode='trilinear'` interpolation. nnU-Net's 2D configurations would build the
  encoder with `nn.Conv2d` and fail — use the `3d_fullres` configurations.
- **Channel constraint for CSIB.** The shared branch width is
  `max(1, bottleneck_channels // n_stages)` and the fused width is that value
  times `n_stages`. `ChannelAttention3D` uses a squeeze ratio of 16, so keep the
  fused width comfortably above 16 (nnU-Net's default 3D plans give 6 stages
  with 320 bottleneck channels → branch width 53, fused width 318).
- **BRH placement.** The head runs at the **full-resolution** decoder stage.
  It uses depthwise convolutions, so its parameter count is small, but its
  activation memory at full resolution is not — this is the most memory-hungry
  insertion point. Attaching it to a coarser stage is possible by changing the
  index in step 3, at the cost of a smaller effective receptive field relative
  to the input voxels.
- **Deep supervision.** BRH is applied *before* `seg_layers`, so with
  `deep_supervision=True` the highest-resolution prediction is refined while the
  lower-resolution auxiliary outputs are left unchanged.
- **Feature-map footprint.** `compute_conv_feature_map_size` in the encoder is
  inherited from the stock implementation and does not account for the extra
  bottleneck tensors built inside CSIB; the effect on the reported value is
  negligible at this resolution.
- **Kernel sizes live in the plans file, not in the code.** If step 6 is skipped
  you silently train a network with stock `3x3x3` kernels that merely has CSIB
  and BRH bolted on — the blocks themselves do not change any kernel size.
- **GPU memory.** The `7x7x7` kernels of stages 2–5, BRH's full-resolution
  activation maps and CSIB's extra bottleneck tensors all increase the footprint
  relative to the stock network. nnU-Net's planner derived its patch size *and*
  its `batch_size` for stock `3x3x3` kernels, so its memory estimate is only a
  lower bound — expect to reduce `configuration_manager.batch_size` in the
  trainer if you run out of memory. The patch size itself can stay as planned,
  since CSIB only consumes existing skips.

---

## 6. Citation

If you use these modules in your research, please cite the paper:

**Xie, S., Lin, C., Huang, D., Zhang, L., & Cai, B. (2027). A panoramically flattened and receptive-field-enhanced framework for topology-preserving mandibular canal segmentation in CBCT. *Expert Systems With Applications*, 333, 134269. https://doi.org/10.1016/j.eswa.2026.134269**

Or in BibTeX format:

```bibtex
@article{xie2027panoramically,
  title   = {A panoramically flattened and receptive-field-enhanced framework for topology-preserving mandibular canal segmentation in CBCT},
  author  = {Xie, Shaojun and Lin, Chaoying and Huang, Dingming and Zhang, Lan and Cai, Biao},
  journal = {Expert Systems with Applications},
  volume  = {333},
  pages   = {134269},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134269}
}
```
