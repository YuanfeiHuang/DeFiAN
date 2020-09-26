# DeFiAN: Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution
This repository is for DeFiAN introduced in the following paper

Yuanfei Huang, Jie Li, Xinbo Gao*, Yanting Hu and Wen Lu, "Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution", arXiv 2020.

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)

# Introduction
Benefiting from the strong capabilities of deep CNNs for feature representation and nonlinear mapping, deeplearning-based methods have achieved excellent performance in single image super-resolution. However, most existing SR methods depend on the high capacity of networks which is initially designed for visual recognition, and rarely consider the initial intention of super-resolution for detail fidelity. 

Aiming at pursuing this intention, there are two challenging issues to be solved: 

(1) learning appropriate operators which is adaptive to the diverse characteristics of smoothes and details; 

(2) improving the ability of model to preserve the low-frequency smoothes and reconstruct the high-frequency details. 

To solve them, we propose a purposeful and interpretable detail-fidelity attention network to progressively process these smoothes and details in divide-and-conquer manner, which is a novel and specific prospect of image super-resolution for the purpose on improving the detail fidelity, instead of blindly designing or employing
the deep CNNs architectures for merely feature representation in local receptive fields. Particularly, we propose a Hessian filtering for interpretable feature representation which is highprofile for detail inference, a dilated encoder-decoder and a distribution alignment cell to improve the inferred Hessian features in morphological manner and statistical manner respectively. Extensive experiments demonstrate that the proposed methods achieve superior performances over the state-of-the-art methods quantitatively and qualitatively.

# Train

# Test
1. Replace the test dataset path '/mnt/Datasets/Test/'.

2. run 'test.py'.

# Results
![PSNR_SSIM_BI](/Figs/psnr_bi_1.PNG)

# Citation
