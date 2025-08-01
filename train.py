import warnings
warnings.filterwarnings("ignore")

import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse, torch, math
from model import Model
from pathlib import Path  
from torch.utils.data import DataLoader 
import torch.optim as optim
from utils import create_gt
from utils import CoCo_Dataset
from utils import reg_loss, soft_focal_loss, Sign_Loss


def get_args_parser():
    parser = argparse.ArgumentParser('Set detector', add_help=False)
    
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--th', default=36, type=int)
    parser.add_argument('--tw', default=36, type=int)
    
    parser.add_argument('--train_imgPath',  default='/workspace/zhouji/dataSets/MS-CoCo/train2017/')
    parser.add_argument('--val_imgPath',    default='/workspace/zhouji/dataSets/MS-CoCo/val2017/')
    parser.add_argument('--train_file',     default='data/S2/train.csv')
    parser.add_argument('--val_file',       default='data/S2/val.csv')
    parser.add_argument('--snapshot',       default=None)
    # parser.add_argument('--snapshot',       default="./dict/model_epoch_200.pth")

    parser.add_argument('--device',         default='cuda')
    parser.add_argument('--start_epoch',    default=0,   type=int)
    parser.add_argument('--epochs',         default=200, type=int)
    parser.add_argument('--save_interval',  default=25,  type=int)
    parser.add_argument('--val_interval',   default=25,  type=int)
    
    parser.add_argument('--losslog',        default='./dict/log/log.txt')
    parser.add_argument('--output_dir',     default='./dict/', help='path to save model')
    return parser

def TemplatMacthLoss(pred_score, gt_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y, gt_param):  
    mask        = (gt_score != 0).float()
    
    scoreLoss   = soft_focal_loss(pred_score[:, 0, :, :], gt_score)
    signLoss, acc = Sign_Loss(pred_sign[:, 0, :, :], gt_param[:, 0, :, :], mask)
    cosLoss = reg_loss(pred_cos[:, 0, :, :], gt_param[:, 1, :, :], mask)
    
    diff, b = 0, gt_score.shape[0]
    for i in range(b):
        gt_y, gt_x = torch.nonzero(gt_score[i, :, :] == gt_score[i, :, :].max())[0]
        pred_angle = torch.arccos(pred_cos[i, 0, gt_y, gt_x])
        gt_angle   = torch.arccos(gt_param[i, 1, gt_y, gt_x])
        diff += torch.abs(pred_angle - gt_angle) * 180 / math.pi
    diff /= b
    
    scaleLoss_x = reg_loss(pred_scale_x[:, 0, :, :], gt_param[:, 2, :, :], mask)
    scaleLoss_y = reg_loss(pred_scale_y[:, 0, :, :], gt_param[:, 3, :, :], mask)
    
    total_loss = scoreLoss + signLoss + cosLoss + scaleLoss_x + scaleLoss_y
    
    return  total_loss, scoreLoss, signLoss, cosLoss, scaleLoss_x, scaleLoss_y, acc, diff

def val(model, val_loader, device, template_shape):
    model.eval()
    acccuray, anglediff = 0.0, 0.0
    with torch.no_grad():
        for i, (_,images, bboxes, center_y, center_x, scale_y, scale_x, angle) in enumerate(val_loader):
            images = images.to(device, non_blocking=True).float()
            bboxes = bboxes.to(device, non_blocking=True).float()
            torch.cuda.empty_cache()
            
            true_params = torch.stack([center_y, center_x, scale_y, scale_x, angle], dim=1).to(torch.float32)
            
            # sign, cos, scale_x, scale_y
            gt_score, gt_param = create_gt(images, true_params, template_shape)
            gt_param, gt_score = gt_param.to(device), gt_score.to(device)
            
            pred_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y = model(images, bboxes)
            *_, acc, diff = TemplatMacthLoss(pred_score, gt_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y, gt_param)
            
            # 计算准确率
            acccuray   += acc.item()
            anglediff  += diff.item()
            
            if i == 200: break
        
        acccuray /= min((i+1), len(val_loader)); anglediff /= min((i+1), len(val_loader))
        return acccuray, anglediff
        
def train(model, train_loader, optimizer, device, template_shape):
    model.train()
    total_loss, scoreloss, signloss, cosloss, scaleloss_x, scaleloss_y = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    acccuray, anglediff = 0.0, 0.0
    for i, (_,images, bboxes, center_y, center_x, scale_y, scale_x, angle) in enumerate(train_loader):
        images = images.to(device, non_blocking=True).float()
        bboxes = bboxes.to(device, non_blocking=True).float()
        torch.cuda.empty_cache()
        
        true_params = torch.stack([center_y, center_x, scale_y, scale_x, angle], dim=1).to(torch.float32)
        
        # sign, cos, sx, sy
        gt_score, gt_param = create_gt(images, true_params, template_shape)
        gt_param, gt_score = gt_param.to(device), gt_score.to(device)

        optimizer.zero_grad(set_to_none=True) 
        
        pred_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y = model(images, bboxes)

        loss, scoreLoss, signLoss, cosLoss, scaleLoss_x, scaleLoss_y, acc, diff = \
            TemplatMacthLoss(pred_score, gt_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y, gt_param)
        
        loss.backward()
        optimizer.step()
        
        total_loss  += loss.item()
        scoreloss   += scoreLoss.item()
        signloss    += signLoss.item()
        cosloss     += cosLoss.item()
        scaleloss_x += scaleLoss_x.item()
        scaleloss_y += scaleLoss_y.item()
        acccuray    += acc.item()
        anglediff   += diff.item()
        
        if i >= 1000: break

    All_loss = [total_loss, scoreloss, signloss, cosloss, scaleloss_x, scaleloss_y, acccuray, anglediff]
    return [x / min((i+1), len(train_loader)) for x in All_loss]


def main(args):
    if not args.output_dir:
        os.makedirs(args.output_dir)

    device = torch.device(args.device)
    model = Model(args.snapshot).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    template_shape = (args.th, args.tw)
    
    train_dataset = CoCo_Dataset(args.train_imgPath, args.train_file, template_shape)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CoCo_Dataset(args.val_imgPath, args.val_file, template_shape)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"模板形状  :{template_shape}")
    print(f"训练数据集: {len(train_dataset)}")
    print(f"验证数据集: {len(val_dataset)}")
    
    for epoch in range(args.start_epoch, args.epochs):
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), args.output_dir + f'model_epoch_{epoch+1}.pth')
        
        if (epoch+1) % args.val_interval == 0:
            try:
                accuray, angle = val(model, val_loader, device, template_shape)
                with open(args.losslog, 'a') as f:
                    f.write(f'Val Epoch [{epoch+1}/{args.epochs}], '
                            f'Acc: {accuray*100:.2f}%, '
                            f'Angle: {angle:.4f},\n'
                            )
            except:
                ...
                        
        loss, scoreLoss, signLoss, cosLoss, scaleLoss_x, scaleLoss_y, accuray, angle = \
            train(model, train_loader, optimizer, device, template_shape)

        with open(args.losslog, 'a') as f:
            f.write(f'Train Epoch [{epoch+1}/{args.epochs}], '
                        f'all: {loss:.4f}, '
                        f'score: {scoreLoss:.4f}, '
                        f'sign: {signLoss:.4f}, '
                        f'cos: {cosLoss:.4f}, '
                        f'scale_x: {scaleLoss_x:.4f}, '
                        f'scale_y: {scaleLoss_y:.4f}, '
                        f'Acc: {accuray*100:.2f}, '
                        f'angle: {angle:.4f},\n'
                        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
