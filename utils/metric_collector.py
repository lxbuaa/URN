import logging
import math

import torchmetrics
import torch
import easydict
import numpy as np
from sklearn import metrics


class AverageMeter(object):
    """用于单个指标的更新、查看"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(self.sum, self.count)


class MetricCollector(object):
    """管理多个指标"""

    def __init__(self):
        """分割指标"""
        self.metrics = easydict.EasyDict({
            "seg": {  # 分割指标
                "f1_seg": AverageMeter(),
                "mcc_seg": AverageMeter(),
            },
            "cls": {  # 分类指标
                "auc_cls": AverageMeter(),
                "acc_cls": AverageMeter(),
            }
        })
        self.save = easydict.EasyDict({
            "seg": {
                "pred_binary": [],  # 二值化之后的pred
                "pred_raw": [],  # 二值化之前的pred
                "gt": []  # groundtruth
            },
            "cls": {
                "pred_binary": [],
                "pred_raw": [],
                "gt": []
            }
        })
        self.y_pred_list, self.y_true_list = [], []
        self.cls_pred_list, self.cls_true_list = [], []

    def update(self, y_pred, y_true, n=1, metric_type='seg', k=0.5):
        temp_pred_tensor, temp_true_tensor, pred_raw_tensor = tensor_binary(y_pred, y_true, k)
        if metric_type == 'cls':
            self.save[metric_type].pred_binary.extend(list(temp_pred_tensor.detach().cpu().numpy()))
            self.save[metric_type].pred_raw.extend(list(pred_raw_tensor.detach().cpu().numpy()))
            self.save[metric_type].gt.extend(list(temp_true_tensor.detach().cpu().numpy()))
        elif metric_type == 'seg':
            if torch.max(temp_true_tensor) >= 1.0:
                temp_value = calculate_pixel_f1(
                    temp_pred_tensor.detach().cpu().numpy(),
                    temp_true_tensor.detach().cpu().numpy()
                )

                self.metrics.seg.f1_seg.update(temp_value['f1'])
                self.metrics.seg.precision_seg.update(temp_value['precision'])
                self.metrics.seg.recall_seg.update(temp_value['recall'])
                self.metrics.seg.mcc_seg.update(temp_value['mcc'])

    def cal_cls_metric(self):
        image_res_dict = calculate_img_score(self.save.cls.pred_binary, self.save.cls.gt)
        self.metrics.cls.acc_cls.update(image_res_dict.acc)
        self.cal_auc('cls')

    def cal_seg_metric(self):
        rev_pixel_res_dict = calculate_pixel_f1(self.save.seg.pred_binary, 1.0 - self.save.seg.gt),
        pixel_res_dict = calculate_pixel_f1(self.save.seg.pred_binary, self.save.seg.gt)
        self.metrics.seg.f1_seg.update(max(pixel_res_dict.f1, rev_pixel_res_dict.f1))
        self.metrics.seg.mcc_seg.update(max(pixel_res_dict.mcc, rev_pixel_res_dict.mcc))

    def cal_auc(self, metric_type='cls'):
        try:
            auc = metrics.roc_auc_score(
                np.array(self.save[metric_type].gt),
                np.array(self.save[metric_type].pred_raw)
            )
        except ValueError:
            auc = 0.0
        self.metrics[metric_type][f'auc_{metric_type}'].update(auc)

    def show(self, metric_type='seg'):
        temp_dict = {}
        if metric_type == 'cls':
            self.cal_cls_metric()
        for metric_name, metric_value in self.metrics[metric_type].items():
            temp_dict[metric_name] = round(float(metric_value.avg), 4)

        print(temp_dict)
        return temp_dict


def tensor_binary(y_pred_tensor, y_true_tensor, k=0.5):
    temp_pred = ((y_pred_tensor.clone() >= k) * 1.0).int()
    pred_raw = y_pred_tensor.clone().float()
    temp_true = y_true_tensor.clone().int()
    return temp_pred, temp_true, pred_raw


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    res_dict = easydict.EasyDict({
        "acc": acc,
    })
    return res_dict


def calculate_pixel_f1(pd, gt):
    """学MVSS，如果全黑的预测对了，指标置1，如果全白的预测错了，指标置0"""
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        res_dict = easydict.EasyDict({
            "f1": 1.0,
            "mcc": 1.0
        })
        return res_dict
    elif np.max(pd) != np.max(gt) and np.max(pd) == 0:
        res_dict = easydict.EasyDict({
            "f1": 0.0,
            "mcc": 0.0
        })
        return res_dict
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    f1_r = 2 * false_neg / (2 * false_neg + true_pos + true_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    mcc = (true_pos * true_neg - false_pos * false_neg) / \
          (math.sqrt(
              (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)) + 1e-6)
    res_dict = easydict.EasyDict({
        "f1": max(f1, f1_r),
        "precision": precision,
        "recall": recall,
        "mcc": mcc
    })
    return res_dict

