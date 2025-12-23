# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import copy
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from sklearn.metrics import accuracy_score as Acc
from sklearn import metrics
import math
import sys
from typing import Iterable, Optional
from scipy import interpolate
import torch.nn as nn
from sklearn.metrics import roc_curve as Roc

import torch
import h5py, os
import numpy as np
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from tqdm import tqdm
import time
from util.loss import *

# criterion_SupCon = SupConLoss()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, tokens=None, resnet=False, criterion_binary_ood=None):
    model.train(True)
    flag = False
    bs = 128

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    torch.cuda.empty_cache()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # time.sleep(10)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        if isinstance(samples, list):
            flag = True
            samples, ood_samples = samples
            ood_samples = ood_samples.to(device, non_blocking=True)


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # labels = targets
        bs = samples.shape[0]

        # tokens_stack = torch.cat([tokens[targets[i]] for i in range(len(targets))], dim=0)
        digital_targets = targets
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # print("here mix")
            # sys.exit()

        with torch.cuda.amp.autocast():

            if resnet:
                logits = model(samples)
            else:
                if flag:
                    samples = torch.cat((samples, ood_samples), dim=0)
                    all_in_logits, text_features, all_out_logits = model(samples, digital_targets=digital_targets,
                                                      detach_aug=args.detach_aug,
                                                      normal_mask=args.normal_mask, ood_samples=flag)
                    logits = all_in_logits[:bs]

                    logits_ood = all_in_logits[bs:]
                    if args.other_label is not None:
                        target_ood = torch.full((bs,), args.other_label, dtype=torch.long, device=device)
                        loss_ood = criterion(logits_ood, target_ood) * args.ood_loss_weight
                    else:
                        prob = torch.softmax(logits_ood, -1)
                        loss_ood = torch.mean(torch.max(prob, -1)[0]) * args.ood_loss_weight
                    # print(loss_ood)

                    # print(torch.max(prob, -1)[0])
                    # sys.exit()
                    # sys.exit()

                    # ood_energy = torch.logsumexp(logits_ood, -1)
                    # loss_ood = torch.mean(ood_energy) * 0.5

                    # print(ood_energy)
                    # print(ood_energy.shape)
                    # print(logits.shape)
                    # sys.exit()

                    # in_max_logits = torch.max(all_in_logits, dim=1)[0]
                    # out_max_logits = torch.max(all_out_logits, dim=1)[0]
                    #
                    # in_max_logits = in_max_logits.unsqueeze(-1)
                    # out_max_logits = out_max_logits.unsqueeze(-1)
                    #
                    # in_out_logits = torch.cat((in_max_logits, out_max_logits), dim=1)
                    # ood_targets = torch.cat((torch.zeros(bs), torch.ones(bs)), dim=0).type(torch.int64).to(device, non_blocking=True)

                else:
                    all_logits, cls_embeddings = model(samples, digital_targets=digital_targets, detach_aug=args.detach_aug,
                                              normal_mask=args.normal_mask)
                    logits = all_logits

                    target_map = None

                    # logits_map = args.in_domain_map_logits.repeat(all_logits.shape[0], 1)
                    # target_map = args.in_domain_map.repeat(all_logits.shape[0], 1)
                    #
                    # for target_idx in range(digital_targets.shape[0]):
                    #     one_target = digital_targets[target_idx]
                    #     if one_target in args.in_domain_list:
                    #         target_map[target_idx] = 1
                    #     else:
                    #         # logits_map[target_idx][one_target] = 0
                    #         target_map[target_idx][one_target] = 1
                    #         target_map[target_idx][digital_targets.flip(0)[target_idx]] = 1

                    # logits = logits + logits_map

                # print(digital_targets.shape)
                # suploss = criterion_SupCon(text_features, labels=digital_targets)

            if args.loss == 'CE':
                loss = criterion(logits, targets, target_map=target_map)
            else:
                loss = criterion(logits, targets, digital_targets=digital_targets)

        if not flag:

            loss_value = loss.item()
        else:
            # loss_ood = criterion_binary_ood(in_out_logits, ood_targets)
            # loss_ood = torch.tensor(0)

            loss = loss + loss_ood
            # print("sdasd")
            loss_value = loss.item()

        # loss_value_supcon = suploss.item()

        # loss_value = loss_value + loss_value_supcon

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if flag:
            metric_logger.update(loss_ood=loss_ood.item())
        # metric_logger.update(regulizations=regulizations.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if flag:
            loss_ood_value_reduce = misc.all_reduce_mean(loss_ood.item())
        # regulizations_value_reduce = misc.all_reduce_mean(regulizations.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            if flag:
                log_writer.add_scalar('loss_ood', loss_ood_value_reduce, epoch_1000x)
            # log_writer.add_scalar('regulizations', regulizations_value_reduce, epoch_1000x)

            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model_eval = copy.deepcopy(model)
    class_to_idx_au = args.class_to_idx
    class_to_idx = args.class_to_idx_val

    # print(class_to_idx)
    # sys.exit()

    if len(class_to_idx_au) > len(class_to_idx):

        class_to_idx_2 = {}

        idx_to_idx = np.array([0] * len(class_to_idx))

        for cls, idx in class_to_idx.items():
            class_to_idx_2[cls] = class_to_idx_au[cls]
            idx_to_idx[idx] = class_to_idx_au[cls]

        # for name, p in model_eval.named_parameters():
        #     if "head" in name:
        #         print(name)
        #         # p = p[idx_to_idx]
        #         print(p.shape)
        #         # print(p)
        # model_eval.module.head.weight = nn.Parameter(model_eval.module.head.weight[idx_to_idx])
        # try:
        #     model_eval.module.head.bias = nn.Parameter(model_eval.module.head.bias[idx_to_idx])
        # except:
        #     print('np bias')
        model_eval.module.fc.weight = nn.Parameter(model_eval.module.fc.weight[idx_to_idx])
        try:
            model_eval.module.fc.bias = nn.Parameter(model_eval.module.fc.bias[idx_to_idx])
        except:
            print('np bias')

    # sys.exit()

    # switch to evaluation mode
    model_eval.eval()

    flag = False
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]

        bs = images[0].shape[0]
        if isinstance(images, list):
            flag = True
            in_images, out_images = images
            # bs = images.shape[0]
            # ood_targets = torch.cat((torch.zeros(bs), torch.ones(bs)), dim=0).type(torch.int64).to(device,
            #                                                                                        non_blocking=True)
            images = in_images
            # print(images.shape)

            # sys.exit()


        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():

            if 'resnet' in args.model:
                output = model_eval(images)
            else:

                if not flag:
                    output, _ = model_eval(images)
                else:
                    all_in_logits, text_features, all_out_logits = model_eval(images, ood_samples=flag)

                    output = all_in_logits
                    # logits = all_in_logits[:bs]
                    #
                    # in_max_logits = torch.max(all_in_logits, dim=1)[0]
                    # out_max_logits = torch.max(all_out_logits, dim=1)[0]
                    #
                    # in_max_logits = in_max_logits.unsqueeze(-1)
                    # out_max_logits = out_max_logits.unsqueeze(-1)
                    #
                    # output = torch.cat((in_max_logits, out_max_logits), dim=1)
                    #
                    # # target = torch.cat((torch.zeros(bs), torch.ones(bs)), dim=0).type(torch.int64).to(device,
                    # #                                                                                        non_blocking=True)
                    # target = torch.ones(bs).type(torch.int64).to(device,non_blocking=True)
                    # target = torch.cat((torch.zeros(bs), torch.ones(bs)), dim=0).type(torch.int64).to(device,
                    #                                                                                        non_blocking=True)

            loss = criterion(output, target)

        if not flag:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        else:
            acc1 = accuracy(output, target, topk=(1,))
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_shot(cls_num_list, args=None):
    # FIXME here follow TADE
    shot = {}
    cls_num_list = torch.tensor(cls_num_list)
    many_shot = cls_num_list > 100
    few_shot = cls_num_list < 20
    medium_shot = ~many_shot & ~few_shot

    shot['many_shot'] = many_shot
    shot['few_shot'] = few_shot
    shot['medium_shot'] = medium_shot

    return shot


# def get_shot(cls_num_list, args=None):
#     shot = {}
#     cls_num_list = torch.tensor(cls_num_list)
#     many_shot = cls_num_list > 100
#     few_shot = cls_num_list < 20
#     medium_shot = ~many_shot & ~few_shot
#
#     # sys.exit()
#     # path = "/home/szzhao/LT_project/vit_LT/granularity_data/inat_18_v3"
#     shot['many_shot'] = np.load(os.path.join(args.gran_path, 'many_shot.npy'), allow_pickle=True)
#     shot['few_shot'] = np.load(os.path.join(args.gran_path, 'few_shot.npy'), allow_pickle=True)
#     shot['medium_shot'] = ~(shot['many_shot'] | shot['few_shot'])
#
#     return shot


def calibration(preds, labels, confidences, num_bins=15):
    assert (len(confidences) == len(preds))
    assert (len(confidences) == len(labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(labels[selected] == preds[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce}


@torch.no_grad()
def evaluate_all_metric(data_loader, model, device, args):
    model.eval()
    nClasses = len(args.cls_num)

    shot = get_shot(args.cls_num, args=args)

    many_shot = shot['many_shot']
    medium_shot = shot['medium_shot']
    few_shot = shot['few_shot']

    predList = np.array([])
    cfdsList = np.array([])
    grndList = np.array([])

    save_count = 0
    for images, labels in tqdm(data_loader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            if 'resnet' in args.model:
                logits = model(images)
            else:
                logits, _ = model(images)

            print(logits.shape)


            cfds, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

            # save_path = "/home/szzhao/OOD/OOD_ViT/info/preds/" + str(save_count) + ".npy"
            # np.save(save_path, [preds.cpu().numpy(), labels])
            # save_count+=1

            cfds = cfds.detach().squeeze().cpu().numpy()
            preds = preds.detach().squeeze().cpu().numpy()
            cfdsList = np.concatenate((cfdsList, cfds))
            predList = np.concatenate((predList, preds))
            grndList = np.concatenate((grndList, labels))

    results_dict = {
        "cfdsList": cfdsList,
        "predList": predList,
        "grndList": grndList
    }

    np.save('./info/pred_2.npy', results_dict)
    #
    cali = calibration(predList, grndList, cfdsList, num_bins=15)
    ece = cali['expected_calibration_error']
    mce = cali['max_calibration_error']

    cfd_per_class = [0] * nClasses
    pdt_per_class = [0] * nClasses
    rgt_per_class = [0] * nClasses
    acc_per_class = [0] * nClasses
    gts_per_class = [0] * nClasses

    cfd_map = [[0] * nClasses for _ in range(nClasses)]
    cfd_cnt = [[0] * nClasses for _ in range(nClasses)]

    for c, g, p in zip(cfdsList, grndList, predList):
        cfd_map[int(p)][int(g)] += c
        cfd_cnt[int(p)][int(g)] += 1
        gts_per_class[int(g)] += 1
        pdt_per_class[int(p)] += 1
        if g == p:
            cfd_per_class[int(g)] += c
            rgt_per_class[int(g)] += 1

    for i in range(nClasses):
        cnt = rgt_per_class[i]
        if cnt != 0:
            acc_per_class[i] = np.round(cnt / gts_per_class[i] * 100, decimals=2)
            cfd_per_class[i] = np.round(cfd_per_class[i] / cnt * 100, decimals=2)

    for i in range(nClasses):
        for j in range(nClasses):
            if cfd_cnt[i][j] != 0:
                cfd_map[i][j] = cfd_map[i][j] / cfd_cnt[i][j]

    avg_acc = np.sum(rgt_per_class) / np.sum(gts_per_class)
    acc_per_class = np.array(acc_per_class)
    many_shot_acc = acc_per_class[many_shot].mean()
    medium_shot_acc = acc_per_class[medium_shot].mean()
    few_shot_acc = acc_per_class[few_shot].mean()

    pdt_per_class = np.array(pdt_per_class)
    gts_per_class = np.array(gts_per_class)
    cls_num = np.array(args.cls_num)
    q = pdt_per_class / np.sum(pdt_per_class)
    pt = gts_per_class / np.sum(gts_per_class)
    ps = cls_num / np.sum(cls_num)

    pdc_s = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(ps + 1e-6))
    pdc_t = np.sum(pt * np.log(pt + 1e-6) - pt * np.log(q + 1e-6))

    result = {
        'avg_acc': np.round(avg_acc * 100, decimals=2).tolist(),
        'ece': np.round(ece * 100, decimals=2).tolist(),
        'mce': np.round(mce * 100, decimals=2).tolist(),
        'many': np.round(many_shot_acc, decimals=2).tolist(),
        'medium': np.round(medium_shot_acc, decimals=2).tolist(),
        'few': np.round(few_shot_acc, decimals=2).tolist(),
        'pdc': np.round(float(pdc_t / pdc_s), decimals=2)
    }

    print(result)

    return result, np.array(cfd_cnt)


to_np = lambda x: x.detach().cpu().numpy()


def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])


def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])


def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))


def infer_ood(model, test_loader, test_dataset, args):
    fpr = cal_all_metric(model, test_loader, test_dataset, args=args)
    return fpr

def cal_all_metric(model, id_dataset, ood_dataset=None, flag=False, epoch=0, args=None):
    model.eval()
    pred_lis = []
    gt_lis = []

    ind_logits, ind_prob, ind_energy = [], [], []
    if flag:
        ind_ctw, ind_atd = [], []
    res = []
    with torch.no_grad():
        for images, labels in tqdm(id_dataset):

            images = images.to('cuda')
            labels = labels.type(torch.long).view(-1)

            # batch = maybe_dictionarize(batch)
            inputs = images
            labels = labels
            logits, features = model(inputs)

            if logits.shape[1] == 1000:
                logits = logits[:, args.in_domain_list]

            pred_lis += list(torch.argmax(logits, -1).detach().cpu().numpy())
            gt_lis += list(labels.detach().cpu().numpy())

            ind_logits += list(max_logit_score(logits))
            ind_prob += list(msp_score(logits))
            ind_energy += list(energy_score(logits))

        # for name, ood_data in ood_dataset.items():

        ood_data = ood_dataset
        ood_logits, ood_prob, ood_energy = [], [], []
        if flag:
            ood_ctw, ood_atd = [], []
        for images, labels in tqdm(ood_data):
            images = images.to('cuda')
            labels = labels.type(torch.long).view(-1)
            inputs = images
            labels = labels

            logits, features = model(inputs)
            if logits.shape[1] == 1000:
                logits = logits[:, args.in_domain_list]


            ood_logits += list(max_logit_score(logits))
            ood_prob += list(msp_score(logits))
            ood_energy += list(energy_score(logits))

        #### MSP
        auc, MSP_fpr = cal_auc_fpr(ind_prob, ood_prob)
        # res.append([epoch, "MSP", name, auc, fpr])
        #### MaxLogit
        auc, MaxLogit_fpr = cal_auc_fpr(ind_logits, ood_logits)
        # res.append([epoch, "MaxLogit", name, auc, fpr])
        #### Energy
        auc, Energy_fpr = cal_auc_fpr(ind_energy, ood_energy)
        # res.append([epoch, "Energy", name, auc, fpr])

        return [MSP_fpr, MaxLogit_fpr, Energy_fpr]
    # pred_lis = np.array(pred_lis)
    # gt_lis = np.array(gt_lis)
    # acc = Acc(gt_lis, pred_lis)
    #
    # id_lis_epoch = [[epoch, acc]]
    # ood_lis_epoch = res
    # print(id_lis_epoch)
    # for lis in ood_lis_epoch:
    #     print(lis)
    # return id_lis_epoch, ood_lis_epoch


def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr, tpr, thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr


def cal_fpr_recall(ind_conf, ood_conf, tpr=0.95):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    fpr, tpr, thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return fpr, thresh