# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
import sys
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")
#
def load_pretrained_weights(model, pretrained_weights, checkpoint_key, class_to_idx=None):

    if "moe" not in pretrained_weights:

        if urlparse(pretrained_weights).scheme:  # If it looks like an URL
            state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
        else:
            state_dict = torch.load(pretrained_weights, map_location="cpu")

            if "optimizer" in state_dict.keys():
                state_dict = state_dict['model']
    
            # try:
            # del state_dict["head.weight"]
            # del state_dict["head.bias"]

            # del state_dict["head.weight_v"]
            # del state_dict["head.weight_g"]
            # for k in ['head.weight', 'head.bias']:
            #     if k in model and model[k].shape != state_dict[k].shape:
            #         # P.log(f"Removing key {k} from pretrained checkpoint")
            #         del model[k]

        if checkpoint_key is not None and checkpoint_key in state_dict:
            # print('dasda')
            logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)

    # if args.global_pool:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    # else:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        print(msg)
        # sys.exit()

        logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    else:
        if urlparse(pretrained_weights).scheme:  # If it looks like an URL
            state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
        else:

            state_dict = torch.load(pretrained_weights, map_location="cpu")

            if "optimizer" in state_dict.keys():
                state_dict = state_dict['model']

            # try:
            # del state_dict["head.weight"]
            # del state_dict["head.bias"]

            # del state_dict["head.weight_v"]
            # del state_dict["head.weight_g"]
            # for k in ['head.weight', 'head.bias']:
            #     if k in model and model[k].shape != state_dict[k].shape:
            #         # P.log(f"Removing key {k} from pretrained checkpoint")
            #         del model[k]

        if checkpoint_key is not None and checkpoint_key in state_dict:
            # print('dasda')
            logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        class_to_idx_1k = np.load("./info/class_to_idx.npy", allow_pickle=True).item()

        # if len(class_to_idx_1k.keys()) > len(class_to_idx.keys()):
        #     class_to_idx_au = class_to_idx_1k
        #     idx_to_idx = np.array([0] * len(class_to_idx.keys()))
        #     remain_idx = []
        #     for cls, idx in class_to_idx.items():
        #         idx_to_idx[idx] = class_to_idx_au[cls]
        #
        #     for i in range(1000):
        #         if i not in idx_to_idx.tolist():
        #             remain_idx.append(i)
        #     remain_idx = np.array(remain_idx)
        #
        #     if 'head.weight' in state_dict:
        #         state_dict['ood_head.weight'] = state_dict['head.weight'][remain_idx]
        #         state_dict['ood_head.bias'] = state_dict['head.bias'][remain_idx]
        #
        #         state_dict['head.weight'] = state_dict['head.weight'][idx_to_idx]
        #         state_dict['head.bias'] = state_dict['head.bias'][idx_to_idx]

        for name, p in model.named_parameters():
            if "experts" in name:
                print(name)
        # sys.exit()

        for name, p in model.named_parameters():
            print(name)
        # sys.exit()

        for moe_index in range(7):
            moe_file = "run_dinov2_imagenet_moe_vvvv" + str(moe_index)
            # expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/" + moe_file + "/ImageNet-LT/vit_base_patch14_dinov2/" + moe_file + "/9.pth"
            # if moe_index == 1:
            #     expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet_moe_vv1_debug_9/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_moe_vv1_debug_9/10.pth"
            # else:
            expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/" + moe_file + "/ImageNet-LT/vit_base_patch14_dinov2/" + moe_file + "/checkpoint.pth"

            # expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet_baseline_withsmooth/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_baseline_withsmooth/checkpoint.pth"

            state_dict_expert = torch.load(expert_path, map_location="cpu")

            print(expert_path)

            if "optimizer" in state_dict_expert.keys():
                state_dict_expert = state_dict_expert['model']

            for name, p in state_dict_expert.items():
                if "blocks.11" in name:
                    new_name = "experts." + str(moe_index) + "." + name[10:]
                    state_dict[new_name] = state_dict_expert[name]

                if "norm.weight" == name:
                    new_name = "expert_norms." + str(moe_index) + ".weight"
                    state_dict[new_name] = state_dict_expert["norm.weight"]
                    print(name)
                if "norm.bias" == name:
                    new_name = "expert_norms." + str(moe_index) + ".bias"
                    state_dict[new_name] = state_dict_expert["norm.bias"]
                    print(name)
        #
        #     # sys.exit()
        #
        expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet_router_head_1/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_router_head_1/checkpoint.pth"
        state_dict_expert = torch.load(expert_path, map_location="cpu")
        #
        # print(expert_path)
        #
        if "optimizer" in state_dict_expert.keys():
            state_dict_expert = state_dict_expert['model']

        for name, p in state_dict_expert.items():
            if "router.weight" in name:
                state_dict["router.weight"] = state_dict_expert[name]
            if "router.bias" in name:
                state_dict["router.bias"] = state_dict_expert[name]

        msg = model.load_state_dict(state_dict, strict=False)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # print(msg)

        # sys.exit()

        logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

# def load_pretrained_weights(model, pretrained_weights, checkpoint_key, class_to_idx):
#     if urlparse(pretrained_weights).scheme:  # If it looks like an URL
#         state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
#     else:
#
#         state_dict = torch.load(pretrained_weights, map_location="cpu")
#
#         if "optimizer" in state_dict.keys():
#             state_dict = state_dict['model']
#
#         # try:
#         # del state_dict["head.weight"]
#         # del state_dict["head.bias"]
#
#         # del state_dict["head.weight_v"]
#         # del state_dict["head.weight_g"]
#         # for k in ['head.weight', 'head.bias']:
#         #     if k in model and model[k].shape != state_dict[k].shape:
#         #         # P.log(f"Removing key {k} from pretrained checkpoint")
#         #         del model[k]
#
#     if checkpoint_key is not None and checkpoint_key in state_dict:
#         # print('dasda')
#         logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
#         state_dict = state_dict[checkpoint_key]
#     # remove `module.` prefix
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#
#     class_to_idx_1k = np.load("./info/class_to_idx.npy", allow_pickle=True).item()
#
#     # if len(class_to_idx_1k.keys()) > len(class_to_idx.keys()):
#     #     class_to_idx_au = class_to_idx_1k
#     #     idx_to_idx = np.array([0] * len(class_to_idx.keys()))
#     #     remain_idx = []
#     #     for cls, idx in class_to_idx.items():
#     #         idx_to_idx[idx] = class_to_idx_au[cls]
#     #
#     #     for i in range(1000):
#     #         if i not in idx_to_idx.tolist():
#     #             remain_idx.append(i)
#     #     remain_idx = np.array(remain_idx)
#     #
#     #     if 'head.weight' in state_dict:
#     #         state_dict['ood_head.weight'] = state_dict['head.weight'][remain_idx]
#     #         state_dict['ood_head.bias'] = state_dict['head.bias'][remain_idx]
#     #
#     #         state_dict['head.weight'] = state_dict['head.weight'][idx_to_idx]
#     #         state_dict['head.bias'] = state_dict['head.bias'][idx_to_idx]
#
#     for name, p in model.named_parameters():
#         if "experts" in name:
#             print(name)
#     # sys.exit()
#
#     for name, p in model.named_parameters():
#         print(name)
#     # sys.exit()
#
#     for moe_index in range(4):
#         # moe_file = "run_dinov2_imagenet_moe_vvvv" + str(moe_index)
#         # expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/" + moe_file + "/ImageNet-LT/vit_base_patch14_dinov2/" + moe_file + "/9.pth"
#         # if moe_index == 1:
#         #     expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet_moe_vv1_debug_9/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_moe_vv1_debug_9/10.pth"
#         # else:
#             # expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/" + moe_file + "/ImageNet-LT/vit_base_patch14_dinov2/" + moe_file + "/checkpoint.pth"
#
#         expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet100/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet100/checkpoint.pth"
#
#         state_dict_expert = torch.load(expert_path, map_location="cpu")
#
#         print(expert_path)
#
#         if "optimizer" in state_dict_expert.keys():
#             state_dict_expert = state_dict_expert['model']
#
#         for name, p in state_dict_expert.items():
#             if "blocks.11" in name:
#                 new_name = "experts." + str(moe_index) + "." + name[10:]
#                 state_dict[new_name] = state_dict_expert[name]
#
#             if "norm.weight" == name:
#                 new_name = "expert_norms." + str(moe_index) + ".weight"
#                 state_dict[new_name] = state_dict_expert["norm.weight"]
#                 print(name)
#             if "norm.bias" == name:
#                 new_name = "expert_norms." + str(moe_index) + ".bias"
#                 state_dict[new_name] = state_dict_expert["norm.bias"]
#                 print(name)
#
#         # sys.exit()
#
#     expert_path = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet100/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet100/checkpoint.pth"
#     state_dict_expert = torch.load(expert_path, map_location="cpu")
#
#     print(expert_path)
#
#     if "optimizer" in state_dict_expert.keys():
#         state_dict_expert = state_dict_expert['model']
#
#     for name, p in state_dict_expert.items():
#         if "router.weight" in name:
#             state_dict["router.weight"] = state_dict_expert[name]
#         if "router.bias" in name:
#             state_dict["router.bias"] = state_dict_expert[name]
#
#     msg = model.load_state_dict(state_dict, strict=False)
#
#     # if args.global_pool:
#     #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
#     # else:
#     #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
#
#     print(msg)
#
#     # sys.exit()
#
#     logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
