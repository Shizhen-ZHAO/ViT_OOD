import torchvision
import numpy as np
import sys
import pdb
import logging
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
# import seaborn as sns
# import matplotlib.pyplot as plt
# import faiss
from tqdm import tqdm
from torchvision import datasets,transforms
# from .svhn_loader import SVHN

import PIL

import torchvision.transforms as trn

# import hubconf


from models_extend.dinov2.models import build_model_dinov2

def check_path(path_tobecheck):
    if not os.path.exists(path_tobecheck):
        os.mkdir(path_tobecheck)
    return path_tobecheck

class Dataset_extract_feature(datasets.ImageFolder):

    def get_cls_num(self):
        cls_num = [0] * len(self.classes)
        for img in self.imgs:
            cls_num[img[1]] += 1
        return cls_num

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

def set_ood_loader_in100(out_dataset):

    out_dataset_root = "/mnt/sda/dataset/IN100/out_domain"

    if out_dataset == 'iNaturalist':
        # /////////////// inat ///////////////
        ood_data = \
            Dataset_extract_feature(root=os.path.join(out_dataset_root, out_dataset),
                                             transform=trn.Compose([
                                                 trn.Resize(256),
                                                 trn.CenterCrop(224),
                                                 trn.ToTensor(),
                                                 trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                             ]))
    elif out_dataset == 'Places':
        # /////////////// Places365 ///////////////
        ood_data = Dataset_extract_feature(root=os.path.join(out_dataset_root, out_dataset),
                                    transform=trn.Compose([
                                        trn.Resize(256),
                                        trn.CenterCrop(224),
                                        trn.ToTensor(),
                                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                    ]))


    elif out_dataset == 'SUN':
        # /////////////// sun ///////////////
        ood_data = Dataset_extract_feature(root=os.path.join(out_dataset_root, out_dataset),
                                    transform=trn.Compose([
                                        trn.Resize(256),
                                        trn.CenterCrop(224),
                                        trn.ToTensor(),
                                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                    ]))
    elif out_dataset == 'dtd':
        # /////////////// texture ///////////////
        ood_data = Dataset_extract_feature(root=os.path.join(out_dataset_root, out_dataset, "images"),
                                    transform=trn.Compose([
                                        trn.Resize(256),
                                        trn.CenterCrop(224),
                                        trn.ToTensor(),
                                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                    ]))

    elif out_dataset == "NINCO":
        ood_data = Dataset_extract_feature(root="/mnt/sda/zsz/OOD_hard/NINCO/NINCO_OOD_classes",
                                           transform=trn.Compose([
                                               trn.Resize(256),
                                               trn.CenterCrop(224),
                                               trn.ToTensor(),
                                               trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                           ]))

    testsetout = ood_data
    if len(testsetout) > 10000:
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1, shuffle=True, num_workers=8)
    return testloaderOut




ckpt_input = "/home/szzhao/OOD/OOD_ViT/ckpt/run_dinov2_imagenet_baseline_withsmooth/ImageNet-LT/vit_base_patch14_dinov2/run_dinov2_imagenet_baseline_withsmooth/checkpoint.pth"
print(ckpt_input)
model = build_model_dinov2(num_classes=1000, class_to_idx={}, ckpt=ckpt_input, model='vit_base_patch14_dinov2_moe').cuda()

imagenet100_root = "/mnt/sda/dataset/imagenet1k"
dataset_name = 'in1k'

model_name = 'dinov2_vitb_train_moe_rounter_vvvv_head_1_moe_map3'
root = '/mnt/sda/dataset/ood_feature'

dataset_root = check_path(os.path.join(root, dataset_name))
feature_root = check_path(os.path.join(dataset_root, model_name))
print(feature_root)
print(ckpt_input)

FEATURE_SAVE_PATH = check_path(os.path.join(feature_root, 'indomain_train_withname'))
FEATURE_SAVE_PATH_val = check_path(os.path.join(feature_root, 'indomain_val_withname'))
FEATURE_SAVE_PATH_out = check_path(os.path.join(feature_root, 'outdomain_withname'))
batch_size = 32

def build_transform():

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    return test_transform


# def extract_indomain_in1k():
#
#     # imagenet100_root = "/mnt/sda/dataset/imagenet1k"
#     imagenet100_train = os.path.join(imagenet100_root, 'train')
#     imagenet100_dataset = Dataset_extract_feature(root=imagenet100_train,
#                                                 transform=build_transform())
#
#     idx_to_class = {}
#     class_to_idx = imagenet100_dataset.class_to_idx
#     for clsname, idx in class_to_idx.items():
#         idx_to_class[str(idx)] = clsname
#     code_to_name = code2name()
#     name_list = []
#
#     sampler = torch.utils.data.SequentialSampler(imagenet100_dataset)
#     data_loader = torch.utils.data.DataLoader(
#         imagenet100_dataset, sampler=sampler,
#         batch_size=batch_size,
#         num_workers=8,
#         drop_last=False
#     )
#     curr_data_id = 0
#
#     FEATURE_SAVE_PATH_info = FEATURE_SAVE_PATH + 'info'
#     check_path(FEATURE_SAVE_PATH_info)
#
#
#     for images, labels, paths in tqdm(data_loader):
#         with torch.no_grad():
#             images = images.to('cuda')
#             labels = labels.type(torch.long).view(-1)
#
#             label_list = labels.numpy().tolist()
#             one_name_list =  [code_to_name[idx_to_class[str(one_label)]] for one_label in label_list]
#
#             with torch.no_grad():
#                 features = model(images)[1].cpu().numpy()
#                 # features = model(images).cpu().numpy()
#                 curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH, str(curr_data_id) + '.npy')
#                 curr_names_save_path = os.path.join(FEATURE_SAVE_PATH_info, str(curr_data_id) + '_names' + '.npy')
#
#                 np.save(curr_names_save_path, [one_name_list, list(paths)])
#                 np.save(curr_feature_save_path, features)
#                 curr_data_id += 1

def extract_indomain_in1k_2():

    imagenet100_train = os.path.join(imagenet100_root, 'train')
    imagenet100_dataset = Dataset_extract_feature(root=imagenet100_train,
                                                transform=build_transform())

    idx_to_class = {}
    class_to_idx = imagenet100_dataset.class_to_idx
    for clsname, idx in class_to_idx.items():
        idx_to_class[str(idx)] = clsname
    code_to_name = code2name()
    name_list = []

    sampler = torch.utils.data.SequentialSampler(imagenet100_dataset)
    data_loader = torch.utils.data.DataLoader(
        imagenet100_dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False
    )
    curr_data_id = 0

    FEATURE_SAVE_PATH_info = FEATURE_SAVE_PATH + 'info'
    check_path(FEATURE_SAVE_PATH_info)


    for images, labels, paths in tqdm(data_loader):
        with torch.no_grad():
            images = images.to('cuda')
            labels = labels.type(torch.long).view(-1)

            label_list = labels.numpy().tolist()
            one_name_list =  [code_to_name[idx_to_class[str(one_label)]] for one_label in label_list]
            one_code_list =  [idx_to_class[str(one_label)] for one_label in label_list]

            with torch.no_grad():
                # features = model(images)[1].cpu().numpy()

                moe_index, features = model(images, return_index=True)
                moe_index = moe_index.cpu().numpy()
                features = features.cpu().numpy()

                curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH, str(curr_data_id) + '.npy')
                curr_names_save_path = os.path.join(FEATURE_SAVE_PATH_info, str(curr_data_id) + '_names' + '.npy')

                np.save(curr_names_save_path, [one_name_list, list(paths), label_list, one_code_list, moe_index])
                np.save(curr_feature_save_path, features)
                curr_data_id +=1


def extract_indomain_val_in1k():
    # imagenet100_root = "/mnt/sda/dataset/imagenet1k"
    imagenet100_train = os.path.join(imagenet100_root, 'val')
    imagenet100_dataset = Dataset_extract_feature(root=imagenet100_train,
                                                transform=build_transform())
    idx_to_class = {}
    class_to_idx = imagenet100_dataset.class_to_idx
    for clsname, idx in class_to_idx.items():
        idx_to_class[str(idx)] = clsname
    code_to_name = code2name()

    sampler = torch.utils.data.SequentialSampler(imagenet100_dataset)

    data_loader = torch.utils.data.DataLoader(
        imagenet100_dataset, sampler=sampler,
        batch_size=1,
        num_workers=8,
        drop_last=False
    )
    curr_data_id = 0

    FEATURE_SAVE_PATH_info = FEATURE_SAVE_PATH_val + 'info'
    check_path(FEATURE_SAVE_PATH_info)

    for images, labels, path in tqdm(data_loader):
        with torch.no_grad():
            images = images.to('cuda')
            labels = labels.type(torch.long).view(-1)
            cls_name = code_to_name[idx_to_class[str(labels.item())]]
            with torch.no_grad():
                # features = model(images)[1].cpu().numpy()
                # features = model(images).cpu().numpy()

                moe_index, features = model(images, return_index=True)
                moe_index = moe_index.cpu().numpy()
                features = features.cpu().numpy()

                curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH_val, str(curr_data_id) + '.npy')

                curr_names_save_path = os.path.join(FEATURE_SAVE_PATH_info, str(curr_data_id) + '_names' + '.npy')
                np.save(curr_names_save_path, [cls_name, path[0], moe_index[0]])

                np.save(curr_feature_save_path, features)
                curr_data_id += 1


def extract_outdomain_val_in1k():

    out_dataset_list = ['iNaturalist', 'Places', 'SUN', 'dtd']
    # out_dataset_list = ['dtd']

    # out_dataset_list = ["NINCO"]

    for out_dataset_name in out_dataset_list:
        out_dataset_loader = set_ood_loader_in100(out_dataset_name)

        FEATURE_SAVE_PATH_out_dataset = os.path.join(FEATURE_SAVE_PATH_out, out_dataset_name)
        FEATURE_SAVE_PATH_out_dataset_info = FEATURE_SAVE_PATH_out_dataset + 'info'

        if not os.path.exists(FEATURE_SAVE_PATH_out_dataset):
            os.mkdir(FEATURE_SAVE_PATH_out_dataset)
            os.mkdir(FEATURE_SAVE_PATH_out_dataset_info)

        curr_data_id = 0
        for images, labels, path in tqdm(out_dataset_loader):
            with torch.no_grad():
                images = images.to('cuda')

                moe_index, features = model(images, return_index=True)
                moe_index = moe_index.cpu().numpy()
                features = features.cpu().numpy()

                curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH_out_dataset, str(curr_data_id) + '.npy')
                curr_names_save_path = os.path.join(FEATURE_SAVE_PATH_out_dataset_info, str(curr_data_id) + '_names' + '.npy')

                np.save(curr_names_save_path, [0, path[0], moe_index[0]])
                np.save(curr_feature_save_path, features)
                curr_data_id += 1


def read_text(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()

    name_list = []
    for line in Lines:
        name_list.append(line[:-1])

    return name_list

def code2name():
    id_path = "./process_granularity/imagenet_22k_clsname/imagenet21k_wordnet_ids.txt"
    name = "./process_granularity/imagenet_22k_clsname/imagenet21k_lemmas.txt"

    id_list = read_text(id_path)
    name_list = read_text(name)
    id2name = {}

    for idx in range(len(id_list)):
        id2name[id_list[idx]] = name_list[idx]
    return id2name

# def extract_indomain_val_in1k():
#
#     imagenet100_root = "/mnt/sda/dataset/imagenet1k"
#     imagenet100_train = os.path.join(imagenet100_root, 'val')
#     imagenet100_dataset = torchvision.datasets.ImageFolder(root=imagenet100_train,
#                                                 transform=build_transform())
#
#     sampler = torch.utils.data.SequentialSampler(imagenet100_dataset)
#     data_loader = torch.utils.data.DataLoader(
#         imagenet100_dataset, sampler=sampler,
#         batch_size=batch_size,
#         num_workers=8,
#         drop_last=False
#     )
#     curr_data_id = 0
#     for images, labels in tqdm(data_loader):
#         with torch.no_grad():
#             images = images.to('cuda')
#             labels = labels.type(torch.long).view(-1)
#             with torch.no_grad():
#                 features = model(images)[1].cpu().numpy()
#                 curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH_val, str(curr_data_id) + '.npy')
#                 np.save(curr_feature_save_path, features)
#                 curr_data_id += 1


# def extract_outdomain_val_in100():
#
#     # out_dataset_list = ['iNaturalist', 'Places', 'SUN', 'dtd']
#     out_dataset_list = ['dtd']
#
#     for out_dataset_name in out_dataset_list:
#         out_dataset_loader = set_ood_loader_in100(out_dataset_name)
#
#         FEATURE_SAVE_PATH_out_dataset = os.path.join(FEATURE_SAVE_PATH_out, out_dataset_name)
#         if not os.path.exists(FEATURE_SAVE_PATH_out_dataset):
#             os.mkdir(FEATURE_SAVE_PATH_out_dataset)
#
#         curr_data_id = 0
#         for images, labels in tqdm(out_dataset_loader):
#             with torch.no_grad():
#                 images = images.to('cuda')
#                 features = model(images)[0].cpu().numpy()
#                 curr_feature_save_path = os.path.join(FEATURE_SAVE_PATH_out_dataset, str(curr_data_id) + '.npy')
#                 np.save(curr_feature_save_path, features)
#                 curr_data_id += 1

if __name__ == '__main__':
    # extract_indomain_in1k()
    # extract_indomain_val_in1k()
    # extract_outdomain_val_in1k()

    extract_indomain_in1k_2()
    extract_indomain_val_in1k()
    extract_outdomain_val_in1k()

    # code2name()


