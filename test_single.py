import warnings
warnings.filterwarnings("ignore")

import cv2, torch, math, time
import numpy as np
from model import Model
from utils import CoCo_Dataset, get_center, getIOU
from utils.refine import *

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


th, tw = 36, 36
rootPath = '/workspace/zhouji/dataSets/MS-CoCo/val2017/'
test_file = 'data/S2/val.csv'
model_file ='dict/model.pth'
result_file = 'res/result-tiny.txt'
device = torch.device('cpu')

model = Model(None).to(device)
state_dict = torch.load(model_file, map_location=device)
model.load_state_dict(state_dict)
model.eval()

dataset = CoCo_Dataset(rootPath, test_file, (th, tw))

success_count = 0
max_samples = 1000

locErr, angleErr, scaleXErr, scaleYErr, mIoU = 0.0, 0.0, 0.0, 0.0, 0.0
all_time = 0.0

with open(result_file, 'w', encoding='utf-8') as rf:
    rf.write('===== 批量模板匹配结果 =====\n\n')

    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            print("Processing sample:", i + 1)
            name, image, template, true_y, true_x, true_sy, true_sx, true_r = dataset[i]
            image = image.unsqueeze(0).to(device)
            template = template.unsqueeze(0).to(device)
            
            start_time = time.time()
            pred_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y = model(image, template)
            inference_time = time.time() - start_time

            pred = pred_score[0, 0, :, :]
            max_idx = torch.nonzero(pred == pred.max())[0]
            pre_Y, pre_X = max_idx.cpu().numpy()
            pred_y, pred_x = get_center((pre_Y, pre_X), torch.nonzero(pred > 0.5).cpu().numpy())

            gt_param = np.array([true_x, true_y, true_sx, true_sy, true_r], dtype=np.float32)

            # 计算预测参数
            sign = 1 if pred_sign[0, 0, pre_Y, pre_X] > 0.5 else -1
            pred_r = torch.arccos(pred_cos[0, 0, pre_Y, pre_X].clamp(-1, 1)) * 180 / math.pi
            pred_r = sign * pred_r.item()
            
            # 计算缩放
            pred_sx = pred_scale_x[0, 0, pre_Y, pre_X].item()
            pred_sy = pred_scale_y[0, 0, pre_Y, pre_X].item()
            pred_param = np.array([pred_x, pred_y, pred_sx, pred_sy, pred_r], dtype=np.float32)

            search_img = ((image.squeeze(0) * 0.5 + 0.5)).permute(1, 2, 0).cpu().numpy()
            template_img = ((template.squeeze(0)*0.5+0.5)).permute(1, 2, 0).cpu().numpy()
            
            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            
            # 进行角度细化
            post_start = time.time()
            refined_angle, refine_score = refine_angle_bisection(
                search_img, template_img, pred_x, pred_y, pred_sx, pred_sy, pred_r,
            )
            pred_sx, _ = refine_scale_x(
                search_img, template_img, pred_x, pred_y, pred_sx, pred_sy, refined_angle,
                search_range=0.15, step_size=0.015
            )
            pred_sy, _ = refine_scale_y(
                search_img, template_img, pred_x, pred_y, pred_sx, pred_sy, refined_angle,
                search_range=0.15, step_size=0.015
            )
            post_time = time.time() - post_start

            pos_error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            angle_error = abs(pred_r - true_r)
            refined_angle_error = abs(refined_angle - true_r)
            angle_error = min(angle_error, 360 - angle_error)
            refined_angle_error = min(refined_angle_error, abs(360 - refined_angle_error))
            scale_x_error = abs(pred_sx - true_sx)
            scale_y_error = abs(pred_sy - true_sy)
            
            # 计算IOU
            pred_param = np.array([pred_x, pred_y, pred_sx, pred_sy, refined_angle], dtype=np.float32)
            IOU = getIOU(pred_param, gt_param, th, tw)

            match_success = (pred_score[0, 0, pre_Y, pre_X].item() > 0.01 and IOU > 0.1)
            if match_success:
                success_count += 1
                rf.write(f'样本 {i} -- {name} \n')
                rf.write('状态: 匹配成功(GT vs. Pred)\n')
                rf.write(f'center: ({true_x:.2f},{true_y:.2f}) vs ({pred_x:.2f},{pred_y:.2f}) 误差: {pos_error:.2f}\n')
                rf.write(f'angle: {true_r:.2f}° vs {refined_angle:.2f}° 误差: {refined_angle_error:.2f}°\n')
                rf.write(f'scale_x: {true_sx:.3f} vs {pred_sx:.3f} 误差: {scale_x_error:.3f}\n')
                rf.write(f'scale_y: {true_sy:.3f} vs {pred_sy:.3f} 误差: {scale_y_error:.3f}\n')
                rf.write(f'IOU: {IOU:.3f}\n')
                rf.write(f'Time(ms): {inference_time*1000:.1f} + {post_time*1000:.1f}')
                rf.write(f'={(inference_time+post_time)*1000:.1f}\n')
                
                locErr += pos_error
                angleErr += refined_angle_error
                scaleXErr += scale_x_error
                scaleYErr += scale_y_error
                mIoU += IOU
                all_time += (inference_time + post_time)   
            else:
                rf.write(f'样本 {i}:\n')
                rf.write('状态: 匹配失败\n')
 
            rf.write('----------------------------\n\n')
            
    rf.write('===== 批量模板匹配结果统计 =====\n\n')
    rf.write(f'总样本数: {min(len(dataset), max_samples)}\n')
    rf.write(f'匹配成功样本数: {success_count}\n')
    rf.write(f'匹配成功率: {success_count / min(len(dataset), max_samples) * 100:.2f}%\n')
    if success_count > 0:
        rf.write(f'平均位置误差: {locErr / success_count:.2f}像素\n')
        rf.write(f'平均角度误差: {angleErr / success_count:.2f}°\n')
        rf.write(f'平均缩放X误差: {scaleXErr / success_count:.3f}\n')
        rf.write(f'平均缩放Y误差: {scaleYErr / success_count:.3f}\n')
        rf.write(f'平均IOU: {mIoU / success_count:.3f}\n')
        rf.write(f'平均时间: {all_time / 1000 * min(len(dataset), max_samples):.2f} ms\n')
    else:
        rf.write('没有成功匹配的样本，无法计算平均误差。\n')
