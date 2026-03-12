# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_class_weight, weight_reduce_loss
from mmseg.registry import MODELS
import warnings
import numpy as np
from torchvision.utils import save_image
import os
import time
from torch.cuda.amp import custom_bwd, custom_fwd


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@MODELS.register_module()
class NCELoss(nn.Module):
    def __init__(self,
                 tau = 1.0,
                 loss_weight = 1.0,
                 loss_name='loss_nce',
            ):
        super().__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(1 / 0.1))
        self._loss_name = loss_name

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 2
        N, C = feat.shape
        feat = feat.permute(1, 0).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N).permute(1, 0)
    
    def forward(self,
                preds_T,
                preds_S
                ):
        loss_kd = 0.
        # save_path_iter = os.path.join("/root/RXDistill/mmseg/models/losses/picc/"+ "iter_{}.jpg".format(time.time()))
        norm_hub = []
        for i in range(len(preds_S)):
            N, C, H, W = preds_S[i].shape
            grad = (preds_T[i] - preds_S[i]).detach()
            loss_kd += 0.5 * F.mse_loss(preds_S[i] , grad)
        loss_feature = self.loss_weight * loss_kd
        return loss_feature

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you wa nt this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
