import torchvision.transforms as transforms
import cv2, json, torch, random, csv, os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .get_imgR import rotate_crop
import pandas as pd
    

class CoCo_Dataset(Dataset):
    def __init__(self, imgPath, fileName, tShape):
        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        self.transform_roi = transforms.Compose([
            transforms.Resize(tShape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        self.imgPath   = imgPath
        self.samples   = []
        self.tShape    = tShape
        
        with open(fileName, 'r') as lines:
            for line in lines:
                name, x, y, w, h = line.strip().split(',')
                self.samples.append([
                    name,
                    (int(x), int(y), int(w), int(h))
                ])
    
    def __len__(self):
        return len(self.samples)
    
    # 数据操作
    def __getitem__(self, idx):
        while True:
            name, bbox  = self.samples[idx]
            image       = cv2.imread(self.imgPath + name)
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            x, y, w, h =  [round(v) for v in bbox]
            template   =  image[y:y+h, x:x+w]
             
            # rotate_crop Image
            imgR, corners, angle = rotate_crop(image, bbox)
            if imgR is None:
                idx = random.randint(0, len(self.samples)-1)
                continue

            scale_y    =  h / self.tShape[0]
            scale_x    =  w / self.tShape[1]
            center     = corners.mean(axis=0)  # 计算中心点

            imgR       = self.transform_img(Image.fromarray(imgR))
            template   = self.transform_roi(Image.fromarray(template))
            
            return imgR, template, center[1], center[0], scale_y, scale_x, angle

            template   = self.transform_roi(Image.fromarray(template))
            
            return name, imgR, template, center[1], center[0], scale_y, scale_x, angle
