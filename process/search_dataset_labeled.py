
import json
from tqdm import tqdm


# path = "/home/szzhao/LT_project/vit_LT/process_granularity/bamboo/bamboo.json"
# with open(path) as f:
#     bamboo = json.load(f)
#
# idx = 'n02084071'
# father2child_idx = bamboo['father2child'][idx]
#
# wordnet_id2trained_id_path = "/home/szzhao/LT_project/vit_LT/process_granularity/bamboo/nid2trainid.json"
# with open(wordnet_id2trained_id_path) as f:
#     wordnet_id2trained_id = json.load(f)
#
# trained_id2clasname_path = "/home/szzhao/LT_project/vit_LT/process_granularity/bamboo/trainid2name.json"
# with open(trained_id2clasname_path) as f:
#     trained_id2clasname = json.load(f)
#
# for word_idx in father2child_idx:
#
#     if word_idx in wordnet_id2trained_id:
#         trained_id = wordnet_id2trained_id[word_idx]
#
#         if trained_id in trained_id2clasname:
#             cls_name = trained_id2clasname[trained_id]
#             print(cls_name)
#             print(trained_id)
#             print(word_idx)
#             print()


# From_file=open('/home/szzhao/LT_project/vit_LT/bamboo/cls/meta/extratrain.txt','r')
# count1 = 0
# idx_list = []
#
# for each_line in tqdm(From_file):
#     line = each_line.split(" ")
#     cls_idx = line[0].split("/")[-2]
#
#     # if cls_idx not in idx_list:
#     idx_list.append(cls_idx)
#     # if cls_idx == "93841":
#     #     print(each_line)
# #print(From_file.read())
# From_file.close()
#
#
# print(len(list(set(idx_list))))
# print(count1)
#

# From_file=open('/home/szzhao/LT_project/vit_LT/bamboo/cls/meta/public.train.txt','r')
# count2 = 0
# idx_list2 = []
# try:
#     for each_line in tqdm(From_file):
#         line = each_line.split(" ")
#         # print(line)
#         dataset_name = line[0].split('/')[0]
#
#         if dataset_name not in idx_list2:
#             idx_list2.append(dataset_name)

        # if len(line) == 2:
        #     wordnet_idx = line[1][:-1]
        # elif len(line) == 1:
        #     print(line)
        #     wordnet_idx = line[1][:-1].split(".")[0]


        # idx_list2.append(wordnet_id2trained_id[wordnet_idx])
        # print(each_line)

        # count2 += 1
        # if cls_idx not in idx_list:
        # idx_list.append(cls_idx)
        # if cls_idx == "93841":
        #     print(each_line)
    #print(From_file.read())
# finally:
#     From_file.close()
# print(idx_list2)

#
#
# print(len(list(set(idx_list2))))
# print(count2)