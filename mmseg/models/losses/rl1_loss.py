# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
import matplotlib.pyplot as plt

def el_get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)
    size.append(N)
    label_one_hot = ones.view(*size)[:,:,:,0:14]
    return label_one_hot

def get_one_hot(label, N_cls):
    # label shape: (N, H, W)
    return F.one_hot(label, num_classes=N_cls).permute(0, 1, 2, 3).float()

@MODELS.register_module()
class RegionL1(nn.Module):
    def __init__(self,tau=1.0, loss_weight=1.0, N_cls=None, loss_name='rl1_loss'):
        super(RegionL1, self).__init__()
        self.tau = tau
        self.relu = nn.ReLU(inplace=False)
        self.loss_weight = loss_weight
        self.N_cls = N_cls
        self._loss_name = loss_name

    def forward(self, preds_S, preds_T):
        """Forward computation.
        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        one_hot = get_one_hot(preds_S.argmax(1), self.N_cls)
        one_hot_encoding = one_hot.contiguous().permute(0, 3, 1, 2)
        region_S = preds_S * one_hot_encoding
        region_T = preds_T * one_hot_encoding
        loss = F.l1_loss(region_S, region_T.clone().detach())
        return loss



    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
