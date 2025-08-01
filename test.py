import warnings
warnings.filterwarnings("ignore")

import cv2, torch, math, random
from model import Model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from utils import draw_rotated_bbox, get_center
from utils.refine import *
from utils import getIOU


device = torch.device('cpu')

model = Model('dict/model.pth').to(device).eval()
transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)            
        ])


query_image_path = './res/imageGray.png'
# query_image_path = './res/image.jpg'
template_image_path = './res/templateGray.png'
# template_image_path = './res/template.jpg'
with torch.no_grad():
    image = cv2.imread(query_image_path)
    template = cv2.imread(template_image_path)
    template = cv2.resize(template, (36, 36))
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tmp_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    image = transform_img(Image.fromarray(img_gray)).to(device).unsqueeze(0)
    template = transform_img(Image.fromarray(tmp_gray)).to(device).unsqueeze(0)
        
    pred_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y = model(image, template)
    
    # 计算 center
    pred = pred_score[0, 0, :, :]
    max_idx = torch.nonzero(pred == pred.max())[0]
    pre_Y, pre_X = max_idx.cpu().numpy()
    pred_y, pred_x = get_center((pre_Y, pre_X), torch.nonzero(pred > 0.5).cpu().numpy())
    
    # 计算 角度
    sign = 1 if pred_sign[0, 0, pre_Y, pre_X] > 0.5 else -1
    pred_r = torch.arccos(pred_cos[0, 0, pre_Y, pre_X].clamp(-1, 1)) * 180 / math.pi
    pred_r = sign * pred_r.item()
    
    # 计算缩放
    pred_sx = pred_scale_x[0, 0, pre_Y, pre_X].item()
    pred_sy = pred_scale_y[0, 0, pre_Y, pre_X].item()
    
    search_img = ((image.squeeze(0) * 0.5 + 0.5)).permute(1, 2, 0).cpu().numpy()
    template_img = ((template.squeeze(0)*0.5+0.5)).permute(1, 2, 0).cpu().numpy()
    # 转灰度图
    search_img_gray = cv2.cvtColor(search_img, cv2.COLOR_RGB2GRAY)
    template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    
    # 细化
    pred_r, _ = refine_angle_bisection(
        search_img_gray, template_img_gray, pred_x, pred_y, pred_sx, pred_sy, pred_r,
        initial_range=20,
    )
    pred_sx, _ = refine_scale_x(
        search_img_gray, template_img_gray, pred_x, pred_y, pred_sx, pred_sy, pred_r
    )
    pred_sy, _ = refine_scale_y(
        search_img_gray, template_img_gray, pred_x, pred_y, pred_sx, pred_sy, pred_r
    )
    pred_r, _ = refine_angle(
        search_img_gray, template_img_gray, pred_x, pred_y, pred_sx, pred_sy, pred_r,
        initial_range=5
    )
    
    # 计算IOU
    pred_param = np.array([pred_x, pred_y, pred_sx, pred_sy, pred_r], dtype=np.float32)
    true_param = np.array([47.47,67.61,0.89,1.31,-78.53], dtype=np.float32)
    IoU = getIOU(true_param, pred_param, 36, 36)
    
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=300)
    ax.imshow(search_img)  
    draw_rotated_bbox(ax, pred_param, 'green', 36, 36)
    # draw_rotated_bbox(ax, true_param, 'red', 36, 36)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('res/res.jpg', bbox_inches='tight', pad_inches=0)
    
    print('\t\tcenter\t\t\tscale_x\t\t\tscale_y\t\t\tangle\t\t\tIOU')

    print(f'true:\t\t{true_param[0]:<6.2f} {true_param[1]:<6.2f}', end='\t\t')
    print(f'{true_param[2]:.2f}',end='\t\t\t')
    print(f'{true_param[3]:.2f}', end='\t\t\t')
    print(f'{true_param[4]:.2f}', end='\t\t\t')
    print(1)

    print(f'pre:\t\t{pred_x:<6.2f} {pred_y:<6.2f}', end='\t\t')
    print(f'{pred_sx:.2f}',end='\t\t\t')
    print(f'{pred_sy:.2f}', end='\t\t\t')
    print(f'{pred_r:.2f}', end='\t\t\t')
    print(f'{IoU:.2f}')
