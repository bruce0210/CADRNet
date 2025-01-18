"""
File: utils_enhanced.py
1. FocalLoss（可替代原 CrossEntropyLoss 用于训练）
2. threshold_filter（基于置信度阈值过滤）
3. nms_postprocessing（基于距离的 NMS 去重）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    多分类场景下的简易 FocalLoss。
    gamma: 聚焦因子，越大越关注难分类样本
    alpha: 权重因子
    """
    def __init__(self, gamma=0.5, alpha=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B*N, C] (C=num_class+1, 含背景)
        targets: [B*N], 取值范围 [0, C-1]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确的概率
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def threshold_filter(logits_or_probs, threshold=0.3):
    """
    对网络输出进行基于置信度的过滤：若最大置信度 < threshold，则将预测类别置为0(背景)。
    logits_or_probs: [N, C] (C=num_class+1)
    threshold: float, 置信度阈值
    返回:
      pred_label: [N], 每个点的预测类别
      pred_conf:  [N], 每个点的最大类别置信度
    """
    if logits_or_probs.dim() == 2 and logits_or_probs.size(1) > 1:
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs
    pred_conf, pred_label = probs.max(dim=-1)
    mask_bg = pred_conf < threshold
    pred_label[mask_bg] = 0
    return pred_label, pred_conf


def nms_postprocessing(xy, labels, conf, dist_threshold=3.0):
    """
    基于“坐标+类别相同+距离较近”来抑制重复预测的简易NMS。
    xy: [N, 2]
    labels: [N]
    conf: [N]
    dist_threshold: float，若同类距离 < dist_threshold 则视为重复
    
    返回：要保留的 idx（在传入顺序下）
    """
    # 根据置信度从高到低排序
    sort_idx = torch.argsort(conf, descending=True)
    xy_sorted = xy[sort_idx]
    labels_sorted = labels[sort_idx]
    conf_sorted = conf[sort_idx]

    keep = []
    for i in range(len(xy_sorted)):
        coord_i = xy_sorted[i]
        label_i = labels_sorted[i]
        if label_i == 0:
            # 背景直接跳过
            continue

        suppressed = False
        for j in keep:
            if labels_sorted[j] == label_i:  # 同类别再检查距离
                dist = torch.norm(coord_i - xy_sorted[j], p=2)
                if dist < dist_threshold:
                    suppressed = True
                    break
        if not suppressed:
            keep.append(i)

    keep = torch.LongTensor(keep).to(xy.device)
    # keep 里存的是 sort 后的索引，需要映射回原顺序
    idx_keep = sort_idx[keep]
    return idx_keep
