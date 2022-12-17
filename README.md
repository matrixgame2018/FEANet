# FEANet-pytorch

![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.11.0](https://img.shields.io/badge/PyTorch-1.11.0-blue) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cmx-cross-modal-fusion-for-rgb-x-semantic/thermal-image-segmentation-on-mfn-dataset)](https://paperswithcode.com/paper/feanet-feature-enhanced-attention-network-for)

This is the official pytorch implementation of [FEANet: FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation](https://arxiv.org/abs/2110.08988) (IEEE IROS). Some of the codes are borrowed from [MFNet](https://github.com/haqishen/MFNet-pytorch) and [RTFNet](https://github.com/yuxiangsun/RTFNet).

The current version supports Python>=3.8.10, CUDA>=11.3.0 and PyTorch>=1.11.0, but it should work fine with lower versions of CUDA and PyTorch. 
![fig2.jpg](https://github.com/matrixgame2018/FEANet/blob/main/figures/fig2.jpg)


## Introduction

Extensive experiments on the urban scene dataset demonstrate that our FEANet outperforms other state-of-the-art (SOTA) RGB-T methods in terms of objective metrics and subjective visual comparison (+2.6% in global mAcc and +0.8% in global mIoU). For the 480 × 640 RGB-T test images, our FEANet can run with a real-time speed on an NVIDIA GeForce RTX 2080 Ti card. Please take a look at the[paper](https://arxiv.org/abs/2110.08988).

 
## Dataset
 
The original dataset can be downloaded from the MFNet project [page](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), but you are encouraged to download our preprocessed dataset from [here](http://gofile.me/4jm56/CfukComo1).

## Pretrained weights

The weights used in the paper:

FEANet : https://drive.google.com/file/d/1hT4ah8D3wjB1nlUjhSmCEYxFx_vC78ki/view?usp=sharing

`python run_own_pth.py -dr [data_dir] -d [test] -f best.pth`

## Training

`python train.py -dr [data_dir] -ls 0.03 -b 5 -em 100`


## RESULTS
![result.png](https://github.com/matrixgame2018/FEANet/blob/main/figures/result.png)

## Citation

If you use FEANet in an academic work, please cite:

```
@inproceedings{DBLP:conf/iros/DengFLWYGCHGL21,
  author    = {Fuqin Deng and
               Hua Feng and
               Mingjian Liang and
               Hongmin Wang and
               Yong Yang and
               Yuan Gao and
               Junfeng Chen and
               Junjie Hu and
               Xiyue Guo and
               Tin Lun Lam},
  title     = {FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time
               Semantic Segmentation},
  booktitle = {{IEEE/RSJ} International Conference on Intelligent Robots and Systems,
               {IROS} 2021, Prague, Czech Republic, September 27 - Oct. 1, 2021},
  pages     = {4467--4473},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/IROS51168.2021.9636084},
  doi       = {10.1109/IROS51168.2021.9636084},
  timestamp = {Wed, 22 Dec 2021 12:37:50 +0100},
  biburl    = {https://dblp.org/rec/conf/iros/DengFLWYGCHGL21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Demos
![fig5.png](https://github.com/matrixgame2018/FEANet/blob/main/figures/fig5.PNG)


## Future Work

A High Accuracy benchmark mark will come soon! The mIoU may achieve 60% the first time.

## blog

[FEANet](https://zhuanlan.zhihu.com/p/421925918)

## Update
| Unlabel | Car  | Person | bike | Curve | Car_stop | guardrail | color_cone | bump | mIoU | Trained model Download | arxivs |
| ------- | ---- | ------ | ---- | ----- | -------- | --------- | ---------- | ---- | ---- | ---------------------- | ------ |
| 98.1    | 87.3 | 71.7   | 63.0 | 48.2  | 42.9     | 24.5      | 53.8       | 54.1 | 60.4 | /                    | /      |

## Contact

Hua Feng：1030001866@qq.com

Mingjian Liang: 2443434059@qq.com
