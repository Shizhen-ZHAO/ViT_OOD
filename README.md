# (ICCV'25) – Equipping Vision Foundation Model with Mixture of Experts for
Out-of-Distribution Detection

Official repo for "Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection".

## Environment
```bash
conda create -n vit_ood python=3.8 -y
conda activate vit_ood
pip install -r requirements.txt
# deps example: torch>=1.7, torchvision>=0.8.1, timm==0.3.2, tensorboardX>=2.1
```

## Data
- ImageNet-LT layout: `./data/ImageNet-LT/{train,val}`
- `gran_path` in scripts can point to your dataset root (e.g., `/mnt/sda/dataset/imagenet1k`).

## Training (MoE + Dynamic Mixup)
- Entrypoints: `script/moe_1/finetune_dinov2_imagenet_moe_*.py`


Example (moe_6):
```bash
python script/moe_1/finetune_dinov2_imagenet_moe_6.py
```


Run strictly in this sequence and align each step's output path with the next step's input (paths adjustable inside scripts).

## Notes
- Classifier head keeps full dataset classes (ImageNet-LT = 1000). `code_group` is only for expert routing/sampling and does not trim head dimension.
- Adjust expert groups via `code_group_v3.pickle` or by changing `group_index` in launch scripts.
