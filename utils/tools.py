#coding=utf-8
import cv2, torch, math, random
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image
  
######################################
#             for train
######################################
   

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.Sigmoid) or isinstance(m, nn.Tanh) or isinstance(m, nn.PixelShuffle):
            pass
        else:
            m.initialize()


def gaussian2D(shape, center, sigma, scale_x, scale_y, rotation=0):
    """
    生成二维高斯热图

    参数:
        shape (tuple): 热图画布的尺寸 (高度, 宽度)
        center (tuple): 高斯分布的中心坐标 (x, y)
        sigma (float or torch.Tensor): 高斯分布的基础标准差（决定热力图的大小）
        scale_x (float or torch.Tensor): x方向的缩放系数
        scale_y (float or torch.Tensor): y方向的缩放系数
        rotation (float or torch.Tensor): 旋转角度（度），正值表示逆时针旋转

    返回:
        torch.Tensor: 二维高斯热图
    """
    height, width = shape
    device = center.device  # 获取输入张量的设备
    
    # 将旋转角度转换为张量（如果输入是float）
    if not isinstance(rotation, torch.Tensor):
        rotation = torch.tensor(rotation, device=device)
    
    # 生成网格坐标
    x = torch.arange(width, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')  # 生成网格坐标 (H, W)

    # 平移到中心点
    xx_centered = xx - center[1]
    yy_centered = yy - center[0]

    # 转换为弧度并计算旋转矩阵元素
    theta = torch.deg2rad(-rotation)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # 应用旋转矩阵（等效于坐标旋转-θ角度）
    xx_rot = xx_centered * cos_theta + yy_centered * sin_theta
    yy_rot = -xx_centered * sin_theta + yy_centered * cos_theta

    # 计算各向异性的标准差（主轴: sigma*scale，次轴: sigma）
    sigma_x = sigma * scale_x
    sigma_y = sigma * scale_y

    # 计算高斯分布
    exponent = (xx_rot ** 2) / (2 * sigma_x ** 2) + (yy_rot ** 2) / (2 * sigma_y ** 2)
    gaussian = torch.exp(-exponent)

    gaussian[gaussian < 0.5] = 0
    return gaussian


def create_gt(images, true_params, template_shape):
    # true_params: [center_y, center_x, scale_y, scale_x, angle]
    b, c, h, w = images.shape
    th, tw = template_shape
    gt_score, gt_param = [], []
    for i in range(b):
        angle = true_params[i, 4]
        sign = 1 if angle > 0 else 0
        cos  = np.cos(np.deg2rad(angle))
        scale_y, scale_x = true_params[i, 2:4]
        sigma = 3.2  # 16 / 4
        HSx, HSy = (tw * scale_x) / min(th, tw), (th * scale_y) / min(th, tw)   # 热力图在两个方向上的缩放系数
        heatmap = gaussian2D((h, w), true_params[i, 0:2], sigma, HSx, HSy, angle)
        gt_score.append(heatmap)
        H, W = heatmap.shape
        gt = torch.zeros((4, H, W))
         
        y,x = torch.where(heatmap > 0)
        gt[0, y, x] = sign
        gt[1, y, x] = cos
        gt[2, y, x] = scale_x
        gt[3, y, x] = scale_y

        gt_param.append(gt)

    # [sign, cos, scale_x, scale_y]
    gt_score = torch.stack(gt_score)
    gt_param = torch.stack(gt_param)
    return gt_score, gt_param


if __name__ == "__main__":
    heatmap = gaussian2D(
        shape=(196, 196),
        center=torch.tensor([164.5, 82.8]),  # 亚像素中心
        sigma=4.0,
        scale_x=2,
        scale_y=1,
        rotation=60.0
    )
    save_image(heatmap, 'heatmap.png')
    
    gt_y, gt_x = torch.nonzero(heatmap == heatmap.max())[0]
    print("gt_y: ", gt_y)  # 33
    print("gt_x: ", gt_x)  # 64
    