import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils import weight_init
     

class convnext_v2_tiny(torch.nn.Module):
    def __init__(self):
        model = timm.create_model('convnextv2_tiny', pretrained=False)
        super().__init__()
        self.stem = model.stem          # Stem层
        self.stage1 = model.stages[0] 
        self.norm = nn.BatchNorm2d(96)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.norm(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            # 仅初始化新增的BatchNorm层（跳过从预训练模型继承的层）
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, snapshot):
        super(Model, self).__init__()
        self.snapshot       =   snapshot
        self.c              =   96
        self.backbone_T     =   convnext_v2_tiny()
        self.backbone_O     =   convnext_v2_tiny()
        self.pointwise_conv =   nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=3, padding=1)
        self.conv1          =   nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=3, padding=1)
        self.conv2          =   nn.Conv2d(in_channels=self.c//2, out_channels=self.c, kernel_size=3, padding=1)
        self.upsample       =   nn.PixelShuffle(2)
        # PixelShuffle: 将每个通道的 4 个像素重新排列到空间维度(无学习权重参数)。从而使通道数减少为原来的 1/4，同时空间维度增加为原来的 2 倍。
        self.conv_score     =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                ) 
        self.conv_sign      =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                )
        self.conv_cos       =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=1, kernel_size=1, padding=0),
                                    nn.Tanh()
                                )
        self.conv_scale_x   =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                )
        self.conv_scale_y   =   nn.Sequential(
                                    nn.Conv2d(in_channels=self.c//4, out_channels=self.c//8, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=self.c//8, out_channels=1, kernel_size=1, padding=0),
                                    nn.Sigmoid()
                                )
        self.bn1            =   nn.BatchNorm2d(self.c*2)
        self.bn2            =   nn.BatchNorm2d(self.c)

        self.initialize()

    # W = 224, H = 224, K = 16
    # TW = K*4, TH = K*4
    def forward(self, origin, template):
        # search_area: [B, C, H, W] = [B, 3, 224, 224]
        # template: [B, C, H, W] = [B, 3, 16, 16]
        # convnext 的第一个阶段（stage1）会对输入进行 4 倍下采样（shape = [B, self.c, H/4, W/4]）

        # W=H = 56, K = 4
        stage1_o = self.backbone_O(origin)       # (B, self.c, 56, 56)
        stage1_t = self.backbone_T(template)       # (B, self.c, 4, 4)
                        
        # depthiwise + pointwise(backbone_T更新梯度)
        output = []
        b, c, th, tw = stage1_t.shape
        for i in range(b):
            out = F.conv2d(
                    input=stage1_o[i].unsqueeze(0), # [self.c, 1, 56, 56]
                    weight=stage1_t[i].view(c, 1, th, tw), # 显式重塑权重,[self.c, 1, 8, 8]
                    stride=1, padding=(th//2, tw//2),
                    groups=self.c
                    )
            out = self.pointwise_conv(out)
            output.append(out)
        feature = torch.cat(output, dim=0)
        # print("feature: ", feature.shape)
        
        out_conv1 = F.relu(self.bn1(self.conv1(feature)))         # (B, self.c*2, 49, 49) 
        out_subconv1 = self.upsample(out_conv1)                   # (B, self.c//2, 98, 98)                                                                                      
        out_conv2 = F.relu(self.bn2(self.conv2(out_subconv1)))    # (B, self.c, 98, 98)
        out = self.upsample(out_conv2)                            # (B, self.c//4 , 196 , 196)

        score    = self.conv_score(out)       # (B, 1, 196, 196)
        sign     = self.conv_sign(out)
        cos      = self.conv_cos(out)
        scale_x  = self.conv_scale_x(out) * 1.5 + 0.5       # scale: 0.5 ~ 2
        scale_y  = self.conv_scale_y(out) * 1.5 + 0.5       # scale: 0.5 ~ 2
                
        return score, sign, cos, scale_x, scale_y
        

    def initialize(self):
        if self.snapshot:
            print('load model...')
            self.load_state_dict(torch.load(self.snapshot, map_location=torch.device('cpu')))
        else:
            weight_init(self)
            nn.init.kaiming_normal_(self.pointwise_conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.pointwise_conv.bias, 0)
