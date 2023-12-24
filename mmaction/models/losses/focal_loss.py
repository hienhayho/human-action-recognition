from typing import List, Optional   

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from mmaction.registry import MODELS
from .base import BaseWeightedLoss

@MODELS.register_module()
class FocalLoss(BaseWeightedLoss):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if weight is not None:
            self.weight = (weight).to(torch.device('cuda'))
        else:
            self.weight = None
        self.ignore_index=ignore_index

    def _forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss