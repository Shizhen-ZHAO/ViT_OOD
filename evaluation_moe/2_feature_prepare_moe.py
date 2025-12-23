import os
import shutil
import sys

import numpy as np
from tqdm import tqdm
import pickle

def check_path(path_tobecheck):
    if not os.path.exists(path_tobecheck):
        os.mkdir(path_tobecheck)
    return path_tobecheck


def read_text(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()

    name_list = []
    for line in Lines:
        name_list.append(line[:-1])

    return name_list

def code2name():
    id_path = "/home/szzhao/OOD/OOD_ViT/process_granularity/imagenet_22k_clsname/imagenet21k_wordnet_ids.txt"
    name = "/home/szzhao/OOD/OOD_ViT/process_granularity/imagenet_22k_clsname/imagenet21k_lemmas.txt"

    id_list = read_text(id_path)
    name_list = read_text(name)
    id2name = {}

    for idx in range(len(id_list)):
        id2name[id_list[idx]] = name_list[idx]
    return id2name

def load_pickle(path):
    with open(path, 'rb') as file:
        # Deserialize and retrieve the variable from the file
        loaded_data = pickle.load(file)
    return loaded_data

info_path = "/home/szzhao/OOD/OOD_ViT/info/code_group/code_group_v3.pickle"
codes_group = load_pickle(info_path)

print(codes_group)

for code in codes_group:
    print(len(code))





# sys.exit()


code2moe = {}
name2moe = {}

code2name_ = code2name()

for m_index in range(len(codes_group)):

    codes = codes_group[m_index]

    for code in codes:

        code2moe[code] = str(m_index)
        name2moe[code2name_[code]] = str(m_index)


num = 9

root = "/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_rounter_vvvv_head_1_moe_map3"
# root = "/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_rounter_head_checkpoint_map1_val3"
path = os.path.join(root, "indomain_train_withname")

info_path = path + "info"

# path = "/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe/indomain_train_withname"
# info_path = "/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe/indomain_train_withnameinfo"

new_path = path + "_" + str(num)
new_info_path = info_path + "_" + str(num)

if not os.path.exists(new_path):
    os.mkdir(new_path)
if not os.path.exists(new_info_path):
    os.mkdir(new_info_path)

file_names = os.listdir(path)
curr_data_id = 0
moe_num = {}

for file_name_idx in tqdm(range(len(file_names))):
    file_name = str(file_name_idx) + '.npy'
    file_path = os.path.join(path, file_name)
    features = np.load(file_path)

    """To obtain corresponding Info file related to the features file"""
    file_index = file_name.split('.')[0]
    info_file_name = file_index + '_names.npy'
    info_file_path = os.path.join(info_path, info_file_name)
    infos = np.load(info_file_path)
    # cls_names, img_paths = infos[0], infos[1]
    print(infos)

    cls_names, img_paths, labels, codes, moe_indexes = infos[0], infos[1], infos[2], infos[3], infos[4]


    for feature_idx in range(features.shape[0]):
        feature = features[feature_idx]

        cls_name = cls_names[feature_idx]
        img_path = img_paths[feature_idx]
        label = labels[feature_idx]
        code = codes[feature_idx]

        # print(code)
        # sys.exit()

        # moe_index = int(float(moe_indexes[feature_idx]))
        moe_index = int(name2moe[cls_name])

        # print("train")
        # print(moe_index)

        if str(moe_index) not in moe_num:
            moe_num[str(moe_index)] = 0

        moe_save_root = check_path(os.path.join(new_path, "moe_"+ str(moe_index)))

        curr_feature_save_path = os.path.join(moe_save_root, str(moe_num[str(moe_index)]) + '.npy')

        moe_num[str(moe_index)] += 1

        np.save(curr_feature_save_path, feature)


path = os.path.join(root, "indomain_val_withname")
info_path = path + "info"

new_path = path + "_" + str(num)
new_info_path = info_path + "_" + str(num)

if not os.path.exists(new_path):
    os.mkdir(new_path)
if not os.path.exists(new_info_path):
    os.mkdir(new_info_path)

file_names = os.listdir(path)
curr_data_id = 0
moe_num = {}

for file_name_idx in tqdm(range(len(file_names))):
    file_name = str(file_name_idx) + '.npy'
    file_path = os.path.join(path, file_name)
    features = np.load(file_path)

    """To obtain corresponding Info file related to the features file"""
    file_index = file_name.split('.')[0]
    info_file_name = file_index + '_names.npy'
    info_file_path = os.path.join(info_path, info_file_name)
    infos = np.load(info_file_path)
    # cls_names, img_paths = infos[0], infos[1]

    cls_names, img_paths, moe_indexes = infos[0], infos[1], infos[2]

    # print(cls_names)
    # sys.exit()

    # moe_index = int(float(moe_indexes))

    moe_index = int(name2moe[cls_names])

    print("val")
    print(moe_index)

    if str(moe_index) not in moe_num:
        moe_num[str(moe_index)] = 0

    moe_save_root = check_path(os.path.join(new_path, "moe_"+ str(moe_index)))

    curr_feature_save_path = os.path.join(moe_save_root, str(moe_num[str(moe_index)]) + '.npy')

    moe_num[str(moe_index)] += 1

    np.save(curr_feature_save_path, features)


    # for feature_idx in range(features.shape[0]):
    #     feature = features[feature_idx]
    #
    #     # print(feature.shape)
    #
    #     cls_name = cls_names[feature_idx]
    #     img_path = img_paths[feature_idx]
    #     label = labels[feature_idx]
    #     code = codes[feature_idx]
    #
    #     moe_index = int(float(moe_indexes[feature_idx]))
    #
    #     if str(moe_index) not in moe_num:
    #         moe_num[str(moe_index)] = 0
    #
    #     moe_save_root = check_path(os.path.join(new_path, "moe_"+ str(moe_index)))
    #
    #     curr_feature_save_path = os.path.join(moe_save_root, str(moe_num[str(moe_index)]) + '.npy')
    #
    #     moe_num[str(moe_index)] += 1
    #
    #     np.save(curr_feature_save_path, feature)

