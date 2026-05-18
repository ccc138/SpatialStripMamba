## Spatial-Strip Mamba for multi-source remote sensing image fusion

## Abstract

Multi-source remote sensing image classification aims to fuse hyperspectral and LiDAR data for fine-grained land cover recognition. Convolutional neural networks are constrained by local receptive fields, while Vision Transformers capture long-range dependencies but suffer from quadratic complexity. The Mamba architecture offers linear efficiency, yet its direct application faces two challenges: flattening 2D images disrupts local adjacency, and existing variants lack multi-scale modeling. To address these issues, we propose Spatial Strip Mamba, an end-to-end U-shaped encoder-decoder network. Our Spatial Locality-Aware Mamba (SLAM) module re-injects 2D local inductive bias via channel-wise depthwise convolutional residual paths with negligible parameter increase. Our Decomposed Strip Multi-scale Attention (DSMA) module decomposes large square kernels into asymmetric 1D strip convolutions, building a multi-scale receptive field pyramid and adaptively recalibrating each scale via channel attention. Experiments on Muufl, Trento, and Augsburg show that Spatial Strip Mamba significantly outperforms state-of-the-art methods, achieving 96.39%, 99.49%, and 97.23% respectively. Particularly notable improvements are observed in challenging categories such as road connectivity, vegetation discrimination, and railway extraction, confirming the effectiveness of our designs.

## Requirements:

* Python 3.11
* PyTorch >= 2.10.0
* mamba-ssm >= 2.3.0

## Usage:

python main.py

