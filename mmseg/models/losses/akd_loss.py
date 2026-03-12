# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

class Feature_Pool(nn.Module):
    """将(B, C, H, W)特征压缩到(B, C)的通道嵌入，并做L2归一化。"""
    def __init__(self, dim: int, ratio: int = 16):
        super().__init__()
        assert isinstance(dim, int) and dim > 0
        assert ratio >= 1
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, dim // ratio)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        pooled = (self.gap_pool(x) + self.gmp_pool(x)) * 0.5      # (B, C, 1, 1)
        v = pooled.flatten(1)                                      # (B, C)
        y = self.up(self.act(self.down(v)))                        # (B, C)
        y = y / (y.norm(dim=1, keepdim=True) + 1e-6)               # L2 norm
        return y                                                   # (B, C)

@MODELS.register_module()
class AKDLoss(nn.Module):
    """
    完全懒初始化版 AKD：
    - 不在 __init__ 里传 dims
    - 第一次 forward 时依据 preds_S 的通道数自动创建每层 Feature_Pool
    - 使用逐通道权重 alpha = sigmoid((w_s * w_t)/tau) 做加权 MSE 蒸馏
    """
    def __init__(self,
                 tau: float = 1.0,
                 loss_weight: float = 1.0,
                 ratio: int = 16,
                 loss_name: str = 'loss_akd'):
        super().__init__()
        self.tau = float(tau)
        self.loss_weight = float(loss_weight)
        self.ratio = int(ratio)
        self._loss_name = loss_name

        self.pools = None
        self._lazy_built = False

    def _lazy_build(self, preds_S):
        """根据学生各层通道数一次性构建 Feature_Pool 列表，并迁移到正确设备。"""
        dims = [p.shape[1] for p in preds_S]
        dev = preds_S[0].device
        self.pools = nn.ModuleList([Feature_Pool(c, self.ratio).to(dev) for c in dims])
        self._built = True

    def forward(self, preds_T, preds_S):
        """
        preds_T / preds_S: list[Tensor]，每个元素形状 (N, C, H, W)，两者长度与对应层的 C 必须一致。
        """
        assert isinstance(preds_T, (list, tuple)) and isinstance(preds_S, (list, tuple)), \
            "preds_T/preds_S 应为 list/tuple[Tensor]"
        assert len(preds_T) == len(preds_S), "Teacher/Student 层数不一致"

        if not self._lazy_built or self.pools is None:
            self._lazy_build(preds_S)

        losses = []
        for t, s, pool in zip(preds_T, preds_S, self.pools):
            assert s.dim() == 4 and t.dim() == 4, "每层应为(N,C,H,W)"
            assert s.shape[1] == t.shape[1], "Teacher/Student 通道数不一致"
            w_s = pool(s)
            w_t = pool(t)
            alpha = torch.sigmoid(s.shape[1] * (w_s * w_t))[:, :, None, None]
            loss_i = F.mse_loss(s * alpha, t * alpha)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
