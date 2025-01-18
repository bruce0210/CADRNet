import os
from tqdm import tqdm
from utils.utils_model import get_pred_instance
from config import anno_config
import torch
from pdb import set_trace as st

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1


# == 新增：从 utils_enhanced.py 文件中导入 threshold_filter, nms_postprocessing ==
from utils.utils_enhanced import threshold_filter, nms_postprocessing
import torch.nn.functional as F

def do_eval(model, dataloader, logger, cfg):
    logger.info(f'> Conducting do_eval')
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在的设备

    anno_list = anno_config.AnnoList().anno_list_all_reverse
    class_num = len(anno_list)
    
    # 初始化计数器，确保在正确的设备上
    cnt_prd = torch.zeros(class_num + 1, device=device)
    cnt_gt = torch.zeros(class_num + 1, device=device)
    cnt_tp = torch.zeros(class_num + 1, device=device)
    
    with torch.no_grad():
        for i, (image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename) in enumerate(tqdm(dataloader, total=len(dataloader), smoothing=0.9)):
            # 将所有输入数据移动到设备上
            image = image.to(device)
            xy = xy.to(device)
            target = target.to(device)
            rgb_info = rgb_info.to(device)
            nns = nns.to(device)
            offset_gt = offset_gt.to(device)
            inst_gt = inst_gt.to(device)
            
            # 前向传播
            seg_pred = model(image, xy, rgb_info, nns)
            seg_pred = seg_pred.contiguous().view(-1, cfg.num_class + 1)
            target = target.view(-1)

            # == 使用基于置信度阈值的过滤 ==
            pred_label, pred_conf = threshold_filter(seg_pred, threshold=0.4)
            
            # == 进一步进行基于距离的 NMS 去重 ==
            xy_flat = xy.view(-1, 2)
            idx_keep = nms_postprocessing(xy_flat, pred_label, pred_conf, dist_threshold=5.0)
            # 对未保留索引的预测设为 0(背景)
            nms_labels = torch.zeros_like(pred_label)
            nms_labels[idx_keep] = pred_label[idx_keep]

            # 统计
            for prd, gt in zip(nms_labels, target):
                cnt_prd[prd] += 1
                cnt_gt[gt] += 1
                if prd == gt:
                    cnt_tp[gt] += 1

    # 计算每个类别的指标
    for cls_id in range(class_num + 1):
        cls_name = "BG" if cls_id == 0 else anno_list[cls_id]
        pred_num = cnt_prd[cls_id].item()
        gt_num = cnt_gt[cls_id].item()
        tp_num = cnt_tp[cls_id].item()
        
        precision = tp_num / (pred_num + 1e-4)
        recall = tp_num / (gt_num + 1e-4)
        f1 = (2 * precision * recall) / (precision + recall + 1e-4)
        
        logger.info("ID:[{:3d}], CLASS:[{:20s}], Pred Num: [{:0>7}], GT Num: [{:0>7}], F1:[{:.2%}], Precision:[{:.2%}], Recall:[{:.2%}]".format(
            cls_id, cls_name, int(pred_num), int(gt_num), f1, precision, recall))
    
    # 计算所有前景类别的总指标（不包括背景）
    if class_num >= 1:
        tp = cnt_tp[1:].sum().item()
        gt = cnt_gt[1:].sum().item()
        pred = cnt_prd[1:].sum().item()
        
        precision = tp / (pred + 1e-4)
        recall = tp / (gt + 1e-4)
        f1 = (2 * precision * recall) / (precision + recall + 1e-4)
        
        logger.info(f'Total FG Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}')
    else:
        logger.warning('No classes found for FG metrics.')
        f1 = 0.0
    
    return f1


def get_eval_criteria(epoch):
    eval = False
    if epoch < 50:
        if epoch % 5 == 0:
            eval = True
    if 50 < epoch < 1e5:
        if epoch % 5 == 0:
            eval = True
    if epoch == 0 or epoch == 1:
        eval = True
    return eval
