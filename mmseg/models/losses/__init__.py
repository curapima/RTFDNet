# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss,M_CrossEntropyLoss)
from .rl1_loss import RegionL1
from .akd_loss import AKDLoss

__all__ = [
    'accuracy', 'Accuracy','CrossEntropyLoss','RegionL1','AKDLoss','M_CrossEntropyLoss'
]
