import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down1 = nn.Linear(dim, dim // ratio)
        self.down2 = nn.Linear(dim, dim // ratio)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = (self.gap_pool(x) + self.gmp_pool(x)) * 0.5
        y = self.up(self.act(self.down1(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2).view(b, c)
        y = (y / y.norm(dim=1, keepdim=True)).contiguous().view(b, c)
        return y
    
class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, bias=False)

    def forward(self, fea1,fea2):
        x1 = self.conv1(fea1)
        x2 = self.conv2(fea2)
        fuse_map = torch.cat([x1, x2], dim=1)
        fuse_map = torch.softmax(fuse_map,dim=1)
        kv_r, kv_x = fuse_map[:, 0:1, :, :], fuse_map[:, 1:2, :, :]
        x = fea1 * kv_r + fea2 * kv_x
        return x

    
class EAEF_clip(nn.Module):
    def __init__(self, dim):
        super(EAEF_clip, self).__init__()
        self.emb_r = Feature_Pool(dim, ratio=4)
        self.emb_x = Feature_Pool(dim, ratio=4)
        self.logit_scale = nn.Parameter(torch.ones([]) * dim)
    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        x1,x2 = self.emb_r(x1),self.emb_x(x2)
        logits_per = self.logit_scale * torch.mul(x1, x2)
        return logits_per


class local_Feature_Fusion(nn.Module):
    def __init__(self,dim):
        super(local_Feature_Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        #self.fusion = Spatial_Attention(dim)
    def forward(self, logits_per,RGB,X):
        b, c, h, w = RGB.size()
        add_gate = self.sigmoid(-1 * logits_per).contiguous().view(b, c, 1, 1)
        ########### ALign Feature ###############################
        x1 = X * add_gate + RGB
        x2 = RGB * add_gate + X
        ########### Feature fusion ###############################
        #x = self.fusion(x1,x2)
        x=(x1+x2)/2
        return x

