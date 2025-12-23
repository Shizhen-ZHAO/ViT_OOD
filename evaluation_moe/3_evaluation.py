import os
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from evaluation_utils import *
# from sklearn.neighbors import DistanceMetric

# from sklearn.metrics.pairwise import pairwise_kernels
#
# pairwise_kernels(X, Y, metric='linear')

from sklearn.metrics.pairwise import linear_kernel
from sklearn.utils.extmath import row_norms, safe_sparse_dot

def dotpro(x, y):
    return safe_sparse_dot(x, y, dense_output=True)

# mydist = DistanceMetric.get_metric('pyfunc', func=dotpro)


# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov1_vitb/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov1_vitb/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov1_vitb/outdomain_withname'


# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/outdomain_withname'

# ID_TRAIN_PATH = '/home/szzhao/ood_feature/clip_baseline/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/home/szzhao/ood_feature/clip_baseline/indomain_val_withname'
# OOD_VAL_PATH = '/home/szzhao/ood_feature/clip_baseline/outdomain_withname'

root = "/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_val_clustervvvv"
model_names = os.listdir(root)
for model_name in model_names:
    ID_TRAIN_PATH = os.path.join(root, model_name, "indomain_train_withname_1")
    ID_VAL_PATH = os.path.join(root, model_name, "indomain_val_withname")
    OOD_VAL_PATH = os.path.join(root, model_name, "outdomain_withname")

# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_dataloader/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_dataloader/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_moe_dataloader/outdomain_withname'


# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_info/outdomain_withname'


# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb/outdomain_withname'

# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_finetuneclstoken/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_finetuneclstoken/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_train_finetuneclstoken/outdomain'

# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_val/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_val/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_val/outdomain_withname'


# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/clip_vitb_trained/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/clip_vitb_trained/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/clip_vitb_trained/outdomain_withname'

# ID_TRAIN_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_lora_30ep_875e6_woaug/indomain_train_withname_1' # dinov2  dinov2_vitb  dinov2_vitb_trained
# ID_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_lora_30ep_875e6_woaug/indomain_val_withname'
# OOD_VAL_PATH = '/mnt/sda/dataset/ood_feature/in1k/dinov2_vitb_all_lora_30ep_875e6_woaug/outdomain_withname'

# ID_TRAIN_PATH = "/home/szzhao/ood_feature/dinov2_vitb_train_cluster_trained_ori/indomain_train_withname_1"
# ID_VAL_PATH = "/home/szzhao/ood_feature/dinov2_vitb_train_cluster_trained_ori/indomain_val_withname"
# OOD_VAL_PATH = "/home/szzhao/ood_feature/dinov2_vitb_train_cluster_trained_ori/outdomain_withname"

    print(ID_TRAIN_PATH)
    print(ID_VAL_PATH)
    print(OOD_VAL_PATH)

# datasets_decs = ["SUN", "Places", "dtd"]
# post_fixs = ["", "_v0", "_v1"]
# set_fixs = ['train', "val"]

# for datasets_dec in datasets_decs:
#     for post_fix in post_fixs:
    # for set_fix in set_fixs:
    # datasets_description = "in1k_" + datasets_dec + post_fix
    # ID_TRAIN_PATH = os.path.join("/mnt/sda/dataset/ood_feature/part", datasets_description, "dinov2_vitb_trained", "indomain_train_withname_1")
    # ID_VAL_PATH = os.path.join("/mnt/sda/dataset/ood_feature/part", datasets_description, "dinov2_vitb_trained", "indomain_val_withname_1")
    # OOD_VAL_PATH = os.path.join("/mnt/sda/dataset/ood_feature/part", datasets_description, "dinov2_vitb_trained", "outdomain_withname")
    #
    # print(ID_TRAIN_PATH)
    # print(ID_VAL_PATH)
    # print(OOD_VAL_PATH)

# OOD_DATASET = ['iNaturalist', 'Places', 'SUN', 'dtd', 'NINCO', "imagenet_o"]
    OOD_DATASET = ['iNaturalist', 'Places', 'SUN', 'dtd']


    ID_TRAIN_RATIO = 0.5
    KNN_K = 300

    VIS = True
    ID_DATASET = 'imagenet1k'
    ENCODER = 'dinov2'
    random.seed(0)

    # load ID train features as knn reference
    id_train_list = os.listdir(ID_TRAIN_PATH)
    id_train_init_num = len(id_train_list)
    id_train_num = int(len(id_train_list) * ID_TRAIN_RATIO)
    random.shuffle(id_train_list)
    id_train_list = id_train_list[:id_train_num]
    print('Build an OOD classifier: ID_DATASET:', ID_DATASET + '(' + str(id_train_init_num) + ')', '   ID_TRAIN_RATIO:', ID_TRAIN_RATIO, '   ID_TRAIN_NUM:', len(id_train_list), '   KNN:', KNN_K, '   Model:', ENCODER)
    id_train_features = []
    for curr_id_file in id_train_list:
        curr_id_path = os.path.join(ID_TRAIN_PATH, curr_id_file)
        curr_feature = np.load(curr_id_path)
        # curr_feature = curr_feature / np.sqrt(np.sum(np.square(curr_feature)))
        id_train_features.append(curr_feature)
    # sys.exit()

    id_train_features = np.array(id_train_features)
    if len(id_train_features.shape) > 2:
        id_train_features = np.squeeze(id_train_features, axis=1)

    print(id_train_features[0].dtype)
    print(f"Total size: {id_train_features[:1000].nbytes / 1024 / 1024:.2f} MB")
    print(f"Total size: {id_train_features.nbytes / 1024 / 1024:.2f} MB")



    # fit a knn for ID train features
    id_train_knn = NearestNeighbors(n_neighbors=KNN_K, algorithm='auto', metric='cosine').fit(id_train_features)
    # id_train_knn = NearestNeighbors(n_neighbors=KNN_K, algorithm='auto', metric=safe_sparse_dot).fit(id_train_features)

    # load ID val features as ID samples
    id_val_list = os.listdir(ID_VAL_PATH)
    id_val_features = []
    for curr_id_file in id_val_list:
        curr_id_path = os.path.join(ID_VAL_PATH, curr_id_file)
        curr_feature = np.load(curr_id_path)
        # curr_feature = curr_feature / np.sqrt(np.sum(np.square(curr_feature)))
        id_val_features.append(curr_feature)
    id_val_features = np.array(id_val_features)
    if len(id_val_features.shape) > 2:
        id_val_features = np.squeeze(id_val_features, axis=1)

    # calculate fpr95 threshold
    id_distances = id_train_knn.kneighbors(id_val_features)[0]
    id_distances_knn = np.max(id_distances, axis=1)

    id_distances_idx = np.argsort(id_distances_knn)
    id_idx_thres = id_distances_idx[int(id_distances_idx.shape[0] * 0.95)]
    id_distances_thres = id_distances_knn[id_idx_thres]
    print(id_distances_thres)

    # load OOD val features as OOD samples and get fpr95 score
    ood_fpr95_scores = []
    ood_auroc_scores = []
    for curr_dataset in OOD_DATASET:
        curr_ood_dataset_path = os.path.join(OOD_VAL_PATH, curr_dataset)
        ood_val_list = os.listdir(curr_ood_dataset_path)
        ood_val_features = []
        for cur_ood_file in ood_val_list:
            curr_ood_path = os.path.join(curr_ood_dataset_path, cur_ood_file)
            curr_feature = np.load(curr_ood_path)
            # curr_feature = curr_feature / np.sqrt(np.sum(np.square(curr_feature)))
            ood_val_features.append(curr_feature)



        ood_val_features = np.array(ood_val_features)
        if len(ood_val_features.shape) > 2:
            ood_val_features = np.squeeze(ood_val_features, axis=1)
        ood_distances = id_train_knn.kneighbors(ood_val_features)[0]
        ood_distances_knn = np.max(ood_distances, axis=1)
        # ood_fpr95_score = np.sum(ood_distances_knn < id_distances_thres) / ood_distances_knn.shape[0]
        # ood_fpr95_scores.append(ood_fpr95_score)
        # print('   OOD:', curr_dataset.ljust(10), 'FPR95:', round(ood_fpr95_score, 4))
        measures = get_measures(-id_distances_knn, -ood_distances_knn, plot=False)
        print('   OOD:', curr_dataset.ljust(15), 'FPR95:', round(measures[2], 4), '      AUROC:', round(measures[1], 4))
        ood_fpr95_scores.append(measures[2])
        ood_auroc_scores.append(measures[1])
        print(1)
    print('   Average:        FPR95:', round(np.mean(np.array(ood_fpr95_scores)), 6), '    AUROC:', round(np.mean(np.array(ood_auroc_scores)), 6))