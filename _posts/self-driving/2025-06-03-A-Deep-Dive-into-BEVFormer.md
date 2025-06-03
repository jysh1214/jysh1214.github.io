---
layout: post
title:  "A Deep Dive into BEVFormer"
date:   2025-06-03
categories: [Self Driving]
author: Alex Chiang
---

## What is BEVFormer? ##

[BEVFormer](https://arxiv.org/pdf/2203.17270) is a camera-only automotive perception algorithm designed for autonomous driving.
In this post, we won’t compare camera-only approaches with LiDAR-based methods. Instead, our focus is exclusively on the architecture and pipeline of `BEVFormer`. This article does not cover training; we’ll focus only on the inference process.

BEV, or Bird’s Eye View, refers to a top-down representation of the environment. It is a widely used format in autonomous driving because it clearly conveys the spatial layout of the scene, including the positions of objects relative to the ego vehicle.

In `BEVFormer`, we use grid-shaped BEV features to represent the surrounding space. These features serve as a unified and structured canvas where multi-view camera information is projected and fused, enabling accurate perception and downstream tasks such as 3D object detection.

**BEVFormer-base Approach**

In this post, we use `BEVFormer-base` [config](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py):
- Input size: `900x1600`
- Backbone: `Resnet101-DCNv2`
- BEV size: `200x200`

In `BEVFormer-base`, the input image size is `900×1600`. However, to ensure that the output feature maps from the backbone and `FPN` are properly aligned and compatible with downsampling operations, the input image height is padded to the nearest multiple of 32. ([source code](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/image/geometric.py#L522-L535))

- The height (900) is not divisible by 32, so it gets padded up to 928.
- The width (1600) is already divisible by 32, so it remains unchanged.

As a result, the image is padded to:
```
900x1600 -> 928x1600
```


## Dataset & Dependencies ##

### nuScenes ###

`BEVFormer` is developed and evaluated on the [nuScenes](https://www.nuscenes.org) dataset, a large-scale benchmark for autonomous driving.
nuScenes provides synchronized data from multiple cameras, LiDAR, radar, and rich metadata, making it well-suited for BEV-based perception research.
`BEVFormer` leverages [CAN bus data](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md) (vehicle speed, steering, etc.) to align historical BEV features.


### OpenMMLab ###

`BEVFormer` is built on top of the [OpenMMLab](https://github.com/open-mmlab) ecosystem, which provides modular, high-performance libraries for computer vision. Understanding these dependencies is essential for setting up `BEVFormer` effectively.

Due to compatibility constraints, `BEVFormer` requires specific package versions. Make sure to install the exact versions listed below to ensure a smooth and successful setup.

**Environment Requirements**
- Python: 3.9.21
- Torch: 1.9.1

**Required Packagies**
- [mmcv-full](https://github.com/open-mmlab/mmcv): 1.4.0
- [mmdet](https://github.com/open-mmlab/mmdetection): 2.14.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d): 0.17.1
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): 0.14.1


## Related Work ##

Before diving into `BEVFormer`, it’s important to first understand several key points.


### BEV Queries ###

We predefine a grid-shaped learnable tensor as the set of queries in `BEVFormer`, where H and W represent the spatial dimensions of the BEV plane. In `BEVFormer-base`, both H and W are set to 200.

$$
Q \in \mathbb{R}^{H \times W \times C}
$$

To specify, the query located at $$p(x, y)$$ of $$Q$$ is responsible for the corresponding grid cell region in the BEV plane.

$$
Q_{p} \in \mathbb{R}^{1 \times C}
$$

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/bev_query.png?raw=true" width="300" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

**BEV Query v.s. BEV Feature**

BEV Query - Learnable weight vectors (embeddings) initialized on the BEV grid; fed into the transformer to gather information.

BEV Feature - The encoded/output feature map in BEV space after fusion of multi-view, multi-scale, and temporal information.


### Bilinear Sampling ###

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/bilinear.png?raw=true" width="450" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

The formula:

$$
f(x, y) = w_{11} f(x_{2}, y_{2}) + w_{12} f(x_{2}, y_{1}) + w_{21} f(x_{1}, y_{2}) + w_{22} f(x_{1}, y_{1})
$$

Where, $$x_{1} = \left \lfloor x \right \rfloor$$, $$x_{2} = x_{1} + 1$$, $$y_{1} = \left \lfloor y \right \rfloor$$, $$y_{2} = y_{1} + 1$$.

### DCN - Deformable Convolutional Network ###

In `BEVFormer-base`, the backbone network is `ResNet-101` with `DCNv2` (deformable convolutional network). This means we use `DCNv2` layers instead of standard `Conv2D` layers in key parts of the network to better capture geometric variations.

**Conv2D v.s. DCN v.s. DCNv2**

`Conv2D`:

$$
y(p) = \sum_{k} x(p_{k}) \cdot w(p_{k})
$$

| Notation   | Description                                                                  |
|------------|------------------------------------------------------------------------------|
| $$p_k$$    | The fixed offset in the convolution kernel (e.g., -1 to 1 for a 3×3 kernel). |
| $$x(p_k)$$ | The input feature at that offset.                                            |
| $$w(p_k)$$ | The learnable weight at that position.                                       |

`DCN` enhances standard convolution by introducing learnable offsets, allowing the network to adapt the sampling locations to the content of the input feature map:

$$
y(p) = \sum_{k} x(p_{k} + \Delta p_{k}) \cdot w(p_{k})
$$

`DCNv2` builds upon `DCN` with two major improvements:
1.	Learned offsets $$\Delta p_k$$ shift the sampling locations dynamically.
2.	Optional modulation masks $$m(p_k)$$ adaptively scale the contribution of each position.

The `DCNv2` formula becomes:

$$
y(p) = \sum_{k} x(p_{k} + \Delta p_{k}) \cdot w(p_{k}) \cdot m(p_{k})
$$

| Notation       | Description                                                            |
|----------------|------------------------------------------------------------------------|
| $$\Delta p_k$$ | Learned offsets - shift the sampling locations dynamically.            |
| $$m(p_k)$$     | Modulation masks - adaptively scale the contribution of each position. |

By using `DCNv2` in the backbone, `BEVFormer` can extract spatially adaptive features that better preserve object shapes and locations, improving the performance of downstream tasks such as BEV transformation and 3D object detection.

The c++ implementation of `DCNv2` in `MMCV` (as modulated deformable convolution) can be found [here](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/parrots/modulated_deform_conv_cpu.cpp#L144-L166).

```cpp
for (int i = 0; i < kernel_h; ++i) {
  for (int j = 0; j < kernel_w; ++j) {
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
        w_col;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    T val = static_cast<T>(0);
    const T h_im = h_in + i * dilation_h + offset_h;
    const T w_im = w_in + j * dilation_w + offset_w;
    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
      val = dmcn_im2col_bilinear_cpu(data_im_ptr, width, height, width,
                                     h_im, w_im);
    *data_col_ptr = val * mask;
    data_col_ptr += batch_size * height_col * width_col;
  }
}
```

Since the learned offsets $$\Delta p_k$$ are often non-integer, feature values at these positions cannot be accessed directly. Instead, bilinear interpolation is used to sample the feature map at non-integer locations.

References:
- [DCN](https://arxiv.org/pdf/1703.06211)
- [DCNv2](https://arxiv.org/pdf/1811.11168)

### Deformable Attention ###

$$
\text{DeformAttn}(Q, V) = \sum_{p=1}^{P} A_p(Q) \cdot V \left(\phi + \Delta p_p(Q) \right)
$$

| Notation          | Description                                                                                                                                                                       |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $$P$$             | Number of sampling points - typically 4 or 8, these are the learned offsets per query.                                                                                            |
| $$Q$$             | Query embedding - a learnable vector representing a position in BEV space (e.g., a cell in a 200x200 grid). It drives the prediction of offsets and attention weights.            |
| $$\phi$$          | Reference points - fixed 2D/3D coordinates associated with each query. It anchors where sampling starts, but does not depend on the query.                                        |
| $$\Delta p_p(Q)$$ | Learned offset - predicted from Q, this defines how far to shift from the reference point for the p-th sampling location. It allows the model to look around the anchor flexibly. |
| $$V(.)$$          | Value — the input tensor to sample from. The sampled values are computed using bilinear interpolation.                                                                            |
| $$A_p(Q)$$        | Attention weight - a query-dependent scalar indicating how important the sampled value is. These are normalized via softmax across P points.                                      |

This equation describes how each query dynamically gathers and fuses information from a set of sampling locations in the input value, where each location is computed by applying learned, query-dependent offsets to a fixed reference point — enabling the model to adaptively focus on geometrically relevant regions.

The following is the [PyTorch CPU implmentation](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/multi_scale_deform_attn.py#L94-L151) in `MMCV`:
```python
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()
```

In practice, deformable attention is designed to work with multi-scale features, such as those produced by a `FPN` (feature pyramid network). This allows the model to sample from feature maps of different spatial resolutions and improves its ability to handle objects at multiple scales.

You may notice in the implementation that `level` and `spatial_shapes` are explicitly included. These are used to manage sampling across different `FPN` levels.

The extended formula for multi-level deformable attention becomes:

$$
\text{DeformAttn}(Q, V) = \sum_{l=0}^{L} \sum_{p=1}^{P} A_p(Q) \cdot V \left(\phi + \Delta p_p(Q) \right)
$$

Where $$L$$ is the number of feature levels (e.g., 4 from the `FPN` in `BEVFormer-base`).

References:
- [CPU implementation](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/multi_scale_deform_attn.py#L94-L151)
- [CUDA implementation](https://github.com/open-mmlab/mmcv/blob/90d83c94cfb967ef162c449faf559616f31f28c2/mmcv/ops/csrc/common/cuda/ms_deform_attn_cuda_kernel.cuh#L200-L254)


## BEVFormer Arch ##

BEVFormer follows a modular architecture that is common in modern vision transformers, consisting of three main parts: Backbone, Neck, and Transformer (Encoder + Decoder).

1. `Backbone`:
Extracts multi-scale image features from input camera images using a deep convolutional neural network, such as `ResNet-101` with modulated deformable convolutions (`DCNv2`).
2. `Neck`:
Aggregates and fuses features from different backbone stages using a Feature Pyramid Network (`FPN`), producing multi-scale feature maps suitable for downstream processing.
3. `Transformer` (Encoder + Decoder):
The core of BEVFormer, this module takes the fused features and processes them in BEV space. The encoder aggregates spatial and temporal context (including information from previous frames), while the decoder generates object queries and refines them into final object detections.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/bevformer_arch.png?raw=true"/>
    <figcaption style="font-size: 14px; color: #555;">(image from paper)</figcaption>
  </figure>
</div>

### Backbone + Neck ###

**Backbone - ResNet101-DCN**

At timestep $$t$$, we feed image features into the backbone network and obtain multi-view feature maps:

$$
F_{t} = \{F_{t}^{i}\}_{i=1}^{N_{\text{view}}}
$$

where $$F_{t}^{i}$$ denotes the feature map from the $$i$$-th camera view, and $$N_{\text{view}}$$ is the total number of camera views. For `BEVFormer-base`, we use 6 surround-view cameras, so $$N_{\text{view}} = 6$$.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/6_cameras.jpg?raw=true"/>
    <figcaption style="font-size: 14px; color: #555;">(image from nuScenes)</figcaption>
  </figure>
</div>

The following images show the visualization of features (6×512×116×200xf32) from `ResNet101`. The last two dimensions represent height and width, and the first dimension corresponds to the 6 input images. Note that I applied post-processing for better visualization.

<div style="text-align: center;">
  <div style="display: inline-block;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_2.png?raw=true" width="200"/>
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_0.png?raw=true" width="200"/>
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_1.png?raw=true" width="200"/>
  </div>
  <br/>
  <div style="display: inline-block;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_4.png?raw=true" width="200"/>
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_3.png?raw=true" width="200"/>
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/8_548s1533151607512404_5.png?raw=true" width="200"/>
  </div>
  <figcaption style="font-size: 14px; color: #555; margin-top: 8px;">
    (image by author)
  </figcaption>
</div>

**Neck - FPN**

In `BEVFormer`, the neck module is implemented using an `FPN` (Feature Pyramid Network), which is designed to extract multi-scale feature maps from the backbone outputs.

- 6x256x116x200xf32 (1/8)
- 6x256x58x100xf32 (1/16)
- 6x256x29x50xf32 (1/32)
- 6x256x15x25xf32 (1/64)

6 is the number of camera views. 256 is the number of channels (feature depth). The last two dimensions represent the scaled (H, W), relative to the padded input size of (928, 1600).


### Encoder ###

The encoder is composed of repeated blocks as defined in the [config](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py#L78-L105).

Components:
- TSA (Temporal Self-Attention)
- Norm
- SCA (Spatial Cross-Attention)
- FFN (Feedforward Network)

The encoder stack follows this sequence, repeated 6 times:
```
(TSA -> Norm -> SCA -> Norm -> FFN -> Norm) * 6
```


### Encoder - TSA ###

`TSA` (temporal self-attention) is a core module in `BEVFormer`’s encoder for modeling temporal dependencies. `TSA` fuses information from the current and previous BEV features, enabling the model to track objects and scene changes over time for robust 3D perception.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/tsa.png?raw=true" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

`TSA` mechanism at each timestamp.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/tsa_history.png?raw=true" width="650"/>
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

**History BEV Feature Alignment**

To ensure spatial consistency, the previous BEV features must be aligned with the current coordinate frame. This alignment is performed using the `can_bus` vector, which encodes the ego vehicle’s motion (e.g., translation and rotation).

The aligned BEV feature at $$t - 1$$ is computed as:

$$
B'_{t - 1}(p) = B_{t - 1}(r \cdot p + \Delta p)
$$

| Symbol            | Description                                                                    |
|-------------------|--------------------------------------------------------------------------------|
| $$B_{t - 1}$$     | The BEV feature map from the previous frame at time $$t - 1$$.                 |
| $$B'_{t - 1}(p)$$ | The BEV feature aligned to the current frame at grid location P.               |
| $$p$$             | A 2D spatial location in the current BEV grid (e.g., a cell in a 200x200 map). |
| $$r$$             | The rotation matrix derived from ego-vehicle.                                  |
| $$\Delta p$$      | The translation offset (shift) in BEV space.                                   |

In practice, we use several steps to align the features:

1. Compute the rotation matrix and shift from the `can_bus` vector. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L122-L141))
2. Rotate the previous BEV feature map. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L143-L156))
3. Add the computed shift to the reference points. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L197-L198))

**Reference Points**

Reference points are fixed 2D coordinates representing each cell in the BEV map (e.g., a 200×200 grid). ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L73-L85)) We add the learnable [offset](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py#L206-L208) to obtain the sampling locations.

The following image is a visualization of fixed 2D points. Note that I adjusted the scale for better visualization.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/fixed_ref2d.png?raw=true" width="450"/>
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

**Attention Weights**

Attention weights are dynamically predicted for each BEV query, determining how much to attend to each sampled image feature point. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py#L209-L217))

**Simplified TSA Formula**

The simplified formula for `TSA` to avoid redundancy between $$p$$ and $$Q_{p}$$:

$$
\text{TSA}(\{ Q, B'_{t - 1} \}) = \sum_{V \in \{ Q, B'_{t - 1} \}} \text{DeformAttn}(Q, V)
$$

| Notation       | Description                                                                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| $$Q$$          | The current BEV queries - learnable embeddings arranged as a 2D grid (e.g., $$200 \times 200 \times 256$$) used as input to the transformer.    |
| $$B'_{t - 1}$$ | The aligned BEV features from the previous frame (at time $$t - 1$$), after rotation and translation using ego-motion (e.g., via CAN bus data). |
| $$V$$          | The input value to sample from. We stack the current BEV queries and aligned history BEV features as the input value.                           |

After applying `DeformAttn`, we fuse the outputs from the history and current values to obtain the final BEV features. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py#L259-L262))

In `TSA`, there is only one feature level (level = 1), as the current and history queries are merged. The spatial shape is set to 200x200 in the config.

If there is no previous frame (e.g., at the first frame), we simply duplicate the query:

$$
\text{TSA}(\{ Q, Q \}) = \sum_{V \in \{ Q, Q \}} \text{DeformAttn}(Q, V)
$$


### Encoder - SCA ###

SCA (spatial cross-attention) is the key module in the `BEVFormer` encoder that fuses multi-view camera features into the BEV (Bird’s Eye View) representation.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/sca.png?raw=true" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

**Flatten Features from FPN**

The multi-scale feature maps from the FPN (feature pyramid network) are flattened at every level.
For example, 4 FPN levels produce feature shapes like: (200 × 116), (100 × 58), (50 × 29), (25 × 15). ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L164-L184))

**Reference Points 3D**

We use a grid of fixed reference points in 3D space (the X–Y plane with fixed Z height). These represent the centers of each BEV query cell. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L60-L71)) In `BEVFormer-base`, the number of sampling points per pillar is [4](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/projects/configs/bevformer/bevformer_base.py#L82).

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/ref_3d.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

To determine which BEV queries are visible to each camera, `BEVFormer` projects 3D reference points into each image plane using the `lidar2img` transformation matrix. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L95-L123)) These reference points are defined at fixed positions in 3D space and correspond to grid locations in the BEV plane. Each of these 3D points is projected into the image, and the model checks whether any of them fall within the image bounds and lie in front of the camera (i.e., within the camera frustum). ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L124-L136)) If at least one projected point is visible, the corresponding BEV cell is considered visible to that camera.

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/lidar2img.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

This visibility check acts as a spatial mask, allowing the model to skip sampling from camera views that do not cover a given BEV region. This improves computational efficiency and ensures that attention is focused on spatially relevant features during the `DeformAttn` operation.

**Corresponding Queries**

For each camera, we project the 3D reference points into its image plane using the `lidar2img` transformation matrix. This tells us which BEV queries are visible (and where) in each camera. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py#L87-L149)) These visible queries are then used as the value input for `DeformAttn`.

The following image shows a visualization of the corresponding queries for cameras 1–6:

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/bev_view.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

**Attention Weights**

Attention weights are dynamically predicted for each BEV query, determining how much to attend to each sampled image feature point. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py#L340-L348))

**Simplified SCA Formula**

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/sca_qf.png?raw=true" width="400" />
    <figcaption style="font-size: 14px; color: #555;">(image by author)</figcaption>
  </figure>
</div>

$$
\text{SCA}(Q, F) = \frac{1}{\left | \nu_{hit} \right |} \sum_{i \in \nu_{hit}} \text{DeformAttn}(Q_{i},F_{i})
$$

| Notation      | Description                                         |
|---------------|-----------------------------------------------------|
| $$Q$$         | BEV Features from the last layer.                   |
| $$F$$         | Flattened multi-scale features from the FPN.        |
| $$\nu_{hit}$$ | The camera view that includes the visible BEV cell. |
| $$i$$         | The index of the $$i$$-th camera.                   |


### Encoder - Norm & FFN ###

`BEVFormer` uses `Norm` (normalization) and `FFN` (feedforward network) layers just like standard transformers: `Norm` ensures training stability and smooth information flow, while `FFN` increases the model’s expressiveness and allows for complex feature transformations at each layer.


### Decoder ###

Configuration details can be found [here](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py#L106-L127).

Components:
- MultiheadAttention
- Norm
- CustomMSDeformableAttention
- FFN

The decoder consists of the following sequence, repeated 6 times:
```
(MultiheadAttention -> Norm -> CustomMSDeformableAttention -> Norm -> FFN -> Norm) * 6
```


### Decoder - MultiheadAttention ###

This layer is a standard multi-head attention module in PyTorch. ([source code](https://github.com/pytorch/pytorch/blob/v1.9.1/torch/nn/modules/activation.py#L873-L1042)) ([source code](https://github.com/pytorch/pytorch/blob/v1.9.1/torch/nn/functional.py#L4836-L5091))

**Multi Head Attention Formula**

For each attention head $$i = 1, …, h$$,

$$
\text{Attention}_{i}(Q_{i}, K_{i}, V_{i}) = \text{softmax} \left ( \frac{Q_{i} K^{T}_{i}}{\sqrt{d_{k}} }  \right ) \cdot V_{i}
$$

The outputs from all heads are concatenated and projected:

$$
\text{AttnOutput} = \text{Concat}(head_{0}, head_{1}, ..., head_{i}) \cdot W^{o}
$$

Here, Q, K, and V are all derived from the learned object query embeddings in the decoder. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py#L136-L137))


### Decoder - CustomMSDeformableAttention ###

The `CustomMSDeformableAttention` layer is a specialized attention mechanism designed to efficiently aggregate information from the BEV features generated by the encoder. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py#L132-L345))

In this layer, the learned object query embeddings serve as queries. These embeddings are used to generate [reference points](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py#L267-L268), [offset](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py#L300-L301), and [attention wights](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/decoder.py#L302-L309).

Finally, `DeformAttn` is applied to sample and aggregate the relevant BEV features from the encoder, allowing each object query to focus adaptively on informative spatial regions.


### Decoder - Norm & FFN ###

`BEVFormer` uses `Norm` (normalization) and `FFN` (feedforward network) layers just like standard transformers: `Norm` ensures training stability and smooth information flow, while `FFN` increases the model’s expressiveness and allows for complex feature transformations at each layer.


### Post-Processing ###

After the decoder produces the output tensors—including class scores, bounding box coordinates, and reference points—BEVFormer performs several post-processing steps to convert these raw outputs into final 3D detection results. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py#L171-L211))

You can see the for-loop by `lvl` variable from `hs`. It's 6 decoder layers here. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py#L108))

> **NOTE**
> Honestly, I don’t know the full name of `hs`. ChatGPT tells me it means “hidden state.”

For each decoder layer (`lvl`), the hidden states are passed through both classification (`cls_branches`) and regression (`reg_branches`) heads to generate per-query class scores and bounding box parameters.

The regression output is refined by adding offsets to reference points, normalized with sigmoid to [0, 1], and then denormalized to real-world coordinates using the configured point cloud range (`pc_range`). These steps transform the model’s relative predictions into absolute 3D bounding box positions in the scene.

**Final Decoding**

During the final decoding stage:

1. Sigmoid activation is applied to class scores to get probabilities. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L54))
2. The top-k predictions are selected, and their corresponding labels and bounding boxes are retrieved. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L55))
3. Bounding box coordinates are denormalized back to real-world values using the point cloud range. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L60))
4. Score thresholding is applied to keep only predictions above a certain confidence level. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L64-L73))
5. Spatial range filtering (post-center range) is performed, so that only bounding boxes whose centers are within a specified 3D region of interest are retained. ([source code](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py#L75-L84))

The final bboxes, scores, and labels are gathered into a prediction dictionary for downstream evaluation or application.

**Visualization**

The BEV:

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/result_bev.jpg?raw=true" width="450" />
  </figure>
</div>

The 3D bboxes predictions in multi-camera images:

<div style="text-align: center;">
  <figure style="display: inline-block; margin: 0;">
    <img src="https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2025-06-03-A-Deep-Dive-into-BEVFormer/result_detect.jpg?raw=true"/>
  </figure>
</div>

## References ##

- [BEVFormer](https://arxiv.org/pdf/2203.17270)
- [BEVFormer GitHub](https://github.com/fundamentalvision/BEVFormer/tree/master)
- [nuScenes](https://www.nuscenes.org)
