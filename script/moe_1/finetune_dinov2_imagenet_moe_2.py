import os
import sys
sys.path.append("..")
sys.path.append(os.getcwd())
from util.trainer import Trainer
from util.trainer import EXP_PATH, WORK_PATH

# def run_dinov2_imagenetlt():
#     T = Trainer()
#     T.task = 'run_dinov2_imagenet_moe'
#     T.note = 'run_dinov2_imagenet_moe'
#
#     T.ckpt = './ckpt/run_dinov2_imagenet/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet/checkpoint.pth'
#     # T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
#
#     T.dataset = 'ImageNet-LT'
#     T.nb_classes = 1000
#
#     T.epochs = 100
#     T.device = '0,1,2,3,4,5,6,7'
#
#     T.batch = 128
#     T.accum_iter = 1
#
#     T.model = 'vit_base_patch14_dinov2'
#     T.input_size = 224
#     T.drop_path = 0.1
#
#     T.clip_grad = None
#     T.weight_decay = 0.05
#     T.adamW2 = 0.99
#
#     T.lr = 0.008
#     T.blr = 8.75e-6
#     T.layer_decay = 0.75
#     T.min_lr = 0.
#     T.warmup_epochs = 10
#     T.clswarm = 0
#
#     T.color_jitter = None
#     T.aa = 'rand-m9-mstd0.5-inc1'
#
#     T.reprob = 0.25
#     T.remode = 'pixel'
#     T.recount = 1
#     T.resplit = False
#
#     T.mixup = 0.8
#     T.cutmix = 1.0
#     T.cutmix_minmax = None
#     T.mixup_prob = 1.0
#     T.mixup_switch_prob = 0.5
#     T.mixup_mode = 'batch'
#
#     T.loss = 'CE'
#     T.bal_tau = 1.0
#     T.smoothing = 0.1
#
#     T.global_pool = True
#
#     T.seed = 0
#     T.prit = 1
#
#     T.lp = 0
#     # T.gran_path = "/mnt/sda/dataset/imagenet1k"
#     # T.gran_path = "/dataset/xinwen/datasets/imagenet"
#     T.gran_path = "/mnt/sda/zsz/data/imagenet_cluster/cluster_0"
#
#     T.num_workers = 16
#     T.master_port = 29521
#
#     T.mask_type = "unmask"
#     # T.ending_block = -1
#
#     T.finetune()
#
# run_dinov2_imagenetlt()

def run_dinov2_imagenetlt():
    T = Trainer()
    T.task = 'run_dinov2_imagenet_moe_vvvv2'
    T.note = 'run_dinov2_imagenet_moe_vvvv2'

    # T.ckpt = './ckpt/run_dinov2_imagenet/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet/checkpoint.pth'
    # T.ckpt = './pretrained_models/dinov2_vitb14_pretrain.pth'
    T.ckpt = "./ckpt/run_dinov2_imagenet_baseline_withsmooth/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_baseline_withsmooth/checkpoint.pth"

    T.dataset = 'ImageNet-LT'
    T.nb_classes = 1000

    T.epochs = 20
    T.device = '0,1,2,3,4,5,6,7'

    T.batch = 128
    T.accum_iter = 1

    T.model = 'vit_base_patch14_dinov2'
    T.input_size = 224
    T.drop_path = 0.1

    T.clip_grad = None
    T.weight_decay = 0.05
    T.adamW2 = 0.99

    T.lr = 0.008
    T.blr = 8.75e-6
    T.layer_decay = 0.75
    T.min_lr = 0.
    T.warmup_epochs = 10
    T.clswarm = 0

    T.color_jitter = None
    T.aa = 'rand-m9-mstd0.5-inc1'

    T.reprob = 0.25
    T.remode = 'pixel'
    T.recount = 1
    T.resplit = False

    T.mixup = 0.0
    T.cutmix = 0.0
    T.cutmix_minmax = None
    T.mixup_prob = 1.0
    T.mixup_switch_prob = 0.5
    T.mixup_mode = 'batch'
    T.dynamic_mixup = True
    T.mixup_alpha_min = 0.1
    T.mixup_alpha_max = 0.1
    T.mixup_difficulty_power = 1.0
    T.dynamic_use_accuracy = False
    T.class_acc_path = ''

    T.loss = 'CE'
    T.bal_tau = 1.0
    T.smoothing = 0.1

    T.global_pool = True

    T.seed = 0
    T.prit = 1

    T.lp = 0
    T.group_file = 'code_group_v3'
    T.gran_path = "/mnt/sda/dataset/imagenet1k"
    # T.gran_path = "/mnt/sdb/szzhao/imagenet1k"
    T.group_index = 2

    T.num_workers = 16
    T.master_port = 29523

    T.mask_type = "unmask"
    # T.ending_block = -1

    T.finetune()

run_dinov2_imagenetlt()