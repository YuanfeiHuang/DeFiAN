# Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution (DeFiAN)
This repository is for DeFiAN introduced in the following paper

Yuanfei Huang, Jie Li, Xinbo Gao*, Yanting Hu and Wen Lu, "Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution", arXiv 2020.
[paper](https://arxiv.org/abs/2009.13134)

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)

## Introduction
the existing SR methods almost consider the information in local receptive fields where only 3x3 kernels of convolution are utilized to represent the local consistent details of feature. However, the collected LR images are full of low-frequency smoothes and high-frequency details, thus, it is natural to raise two issues: (1) It is difficult to learn a perfect convolutional operator, which is adaptive to the diverse characteristics of smoothes and details; (2) How to improve the ability to preserve the low-frequency smoothes and reconstruct the high-frequency details?

(1) For the first issue, since the low-frequency smoothes and high-frequency details have different characteristics of representation, and the ill-posed problem of SR is more sensitive to the fidelity of deficient details, it is better to solve it in a divide-and-conquer manner.

(2) For the second issue, following the first issue, we should preserve the low-frequency smoothes and reconstruct the high-frequency details as better as possible, which aims at reconstructing the residues (in architectures with global residual learning) using detail-fidelity features.

![Framework of DeFiAN](/Figs/Framework_DeFiAN.png)

For these issues and to process the low-frequency smoothes and high-frequency details in a divide-and-conquer manner, we propose a purposeful and interpretable method to improve SR performance using Detail-Fidelity Attention in very deep Networks (DeFiAN). The major contributions of the proposed method are:

(1) Introducing a detail-fidelity attention mechanism in each module of networks to adaptively improve the desired high-frequency details and preserve the low-frequency smoothes in a divide-and-conquer manner, which is purposeful for SISR task.

(2) Proposing a novel multi-scale Hessian filtering (MSHF) to extract the multi-scale textures and details with the maximum eigenvalue of scaled Hessian features implemented using high-profile CNNs. Unlike the conventional CNN features in most existing SR methods, the proposed MSHF is interpretable and specific to improve detail fidelity. Besides, the proposed multi-scale and generic Hessian filtering are the first attempts for interpretable detail inference in SISR task, and could be implemented using GPU-accelerate CNNs without any calculation of intricate inverse of matrix.

(3) Designing a dilated encoder-decoder (DiEnDec) for fusing the full-resolution contextual information of multi-scale Hessian features and inferring the detail-fidelity attention representations in a morphological erosion & dilation manner, which possesses characteristics of both full-resolution and progressively growing receptive fields.

(4) Proposing a learnable distribution alignment cell (DAC) for adaptively expanding and aligning the attention representation under the prior distribution of referenced features in a statistical manner, which is appropriate for residual attention architectures.


## Train

## Test
1. Download models from [OneDrive](https://1drv.ms/u/s!ArdHek-3P6D-avrb8QJPrzqeU2c?e=Md84Tw) or [BaiduYun](https://pan.baidu.com/s/10fLcejD2N5nTnk-TUj9a_A)(password: 3vj9).

2. Replace the test dataset path '/mnt/Datasets/Test/' with your datasets.

3. run 'test.py'.

## Results
### Quantitative Results (PSNR/SSIM)
![Quantitative Results](/Figs/Quantitative_Results.png)

### Qualitative Results
![Fig.6](/Figs/Fig_6.png)
![Fig.7](/Figs/Fig_7.png)
![Fig.8](/Figs/Fig_8.png)
![Fig.9](/Figs/Fig_9.png) 

## Citation
```
@ARTICLE{2020arXiv200913134H,
       author = {{Huang}, Yuanfei and {Li}, Jie and {Gao}, Xinbo and {Hu}, Yanting and
         {Lu}, Wen},
        title = "{Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2020,
        month = sep,
          eid = {arXiv:2009.13134},
        pages = {arXiv:2009.13134},
archivePrefix = {arXiv},
       eprint = {2009.13134},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200913134H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
