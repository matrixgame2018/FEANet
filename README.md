# FEANet-pytorch

This is the official pytorch implementation of [FEANet: FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation](https://github.com/yuxiangsun/RTFNet/blob/master/doc/RAL2019_RTFNet.pdf) (IEEE IROS). Some of the codes are borrowed from [MFNet](https://github.com/haqishen/MFNet-pytorch) and [RTFNet](https://github.com/yuxiangsun/RTFNet).

The current version supports Python>=3.8.10, CUDA>=11.3.0 and PyTorch>=1.11.0, but it should work fine with lower versions of CUDA and PyTorch. 
![fig2.jpg](https://github.com/matrixgame2018/FEANet/tree/main/figures/fig2.png)


## Introduction

Extensive experiments on the urban scene dataset demonstrate that our FEANet outperforms other state-of-the-art (SOTA) RGB-T methods in terms of objective metrics and subjective visual comparison (+2.6% in global mAcc and +0.8% in global mIoU). For the 480 × 640 RGB-T test images, our FEANet can run with a real-time speed on an NVIDIA GeForce RTX 2080 Ti card. Please take a look at the[paper](https://arxiv.org/abs/2110.08988).

 
## Dataset
 
The original dataset can be downloaded from the MFNet project [page](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), but you are encouraged to download our preprocessed dataset from [here](http://gofile.me/4jm56/CfukComo1).

## Pretrained weights

The weights used in the paper:

FEANet : https://gofile.io/d/LAmzOE

`python run_own_pth.py -dr [data_dir] -d [test] -f best.pth`

## Training

`python train.py -dr [data_dir] -ls 0.03 -b 5 -em 100`


## RESULTS
![result.png](https://github.com/matrixgame2018/FEANet/tree/main/figures/result.png)

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
![fig5.png](https://github.com/matrixgame2018/FEANet/tree/main/figures/fig5.png)


## Contact

Mingjian Liang: 2443434059@qq.com
