# (ICCV'25) – Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection


Shizhen Zhao, Jiahui Liu, Xin Wen, Haoru Tan, XiaoJuan Qi

This repository is the official PyTorch implementation of the [paper](https://arxiv.org/abs/2510.10584).



## Environment
```bash
conda create -n vit_ood python=3.8 -y
conda activate vit_ood
pip install -r requirements.txt
# deps example: torch>=1.7, torchvision>=0.8.1, timm==0.3.2, tensorboardX>=2.1
```

## Data
- ImageNet layout: `./data/ImageNet/{train,val}`
- `gran_path` in scripts can point to your dataset root (e.g., `/mnt/sda/dataset/imagenet1k`).

## Training 
- Entrypoints: `script/moe_1/finetune_dinov2_imagenet_moe_*.py`
[Checkpoints](https://drive.google.com/drive/folders/12k54z20p6tG3SsSp3tYmVQysAjF9salg?usp=sharing)

Example (moe_6):
```bash
python script/moe_1/finetune_dinov2_imagenet_moe_6.py
```


## Citation
If you find our idea or code inspiring, please cite our paper:
```bibtex
@InProceedings{Zhao_2025_ICCV,
    author    = {Zhao, Shizhen and Liu, Jiahui and Wen, Xin and Tan, Haoru and Qi, Xiaojuan},
    title     = {Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {1751-1761}
}
```
This code is partially based on [LiVT](https://github.com/XuZhengzhuo/LiVT), if you use our code, please also cite：
```bibtex
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Zhengzhuo and Liu, Ruikang and Yang, Shuo and Chai, Zenghao and Yuan, Chun},
    title     = {Learning Imbalanced Data With Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15793-15803}
}
```

