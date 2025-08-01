import torch
import torch.nn.functional as F


def soft_focal_loss(pred, gt):
    # 限制预测值，防止出现梯度消失/爆炸
    pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6) 
    
    pos_mask = gt.ge(0.5).float() # 大于等于0.5 设为正样本 设置为一个二值化掩码
    neg_mask  = gt.lt(0.5).float() # 小于0.5设为负样本
    
    # 正样本损失
    # pos_loss = F.binary_cross_entropy(pred, gt, reduction="none") * pos_mask
    pos_loss = (gt * torch.log(pred) + (1-gt) * torch.log(1-pred)) * pos_mask
    # 负样本损失 --- mask(1-gt) 全是1
    neg_loss = torch.log(1 - pred) * neg_mask
    
    num_pos = pos_mask.sum().clamp(min=1.0) # 正样本个数
    
    loss = - (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def reg_loss(pred, gt_param, mask):
    regr_loss = F.smooth_l1_loss(pred * mask, gt_param * mask, reduction='sum')
    regr_loss = regr_loss / (mask.sum() + 1e-6)

    return regr_loss 


def Sign_Loss(pred, gt_param, mask):
    acc = ((pred.round() == gt_param)* mask).sum() / (mask.sum() + 1e-6)  # 正确率
    pos_weight = (gt_param * mask).sum() / (mask.sum() + 1e-6)
    neg_weight = 1 - pos_weight
    sign_loss = F.binary_cross_entropy_with_logits(pred * mask, gt_param * mask, reduction='none') 
    # reduction='none'：保持原始形状，不进行均值或求和，以便后续加权。
    
    weighted_loss = sign_loss * (gt_param * pos_weight + (1-gt_param) * neg_weight)
    sign_loss = (weighted_loss * mask).sum() / (mask.sum() + 1e-6) 

    # `Sign_Loss` 的目的是衡量预测值 `pred` 与真实标签 `gt_param` 在符号（正负）上的匹配程度，
    # 最终 `loss` 反映的是模型在符号预测上的误差。
    return sign_loss, acc

