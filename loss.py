#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:00:47 2022

@author: chuhsuanlin
"""


import torch
from torch import nn


class FocalLoss(nn.Module):
    """
    Reference:
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-loss_bce)
        loss_f = self.alpha * (torch.tensor(1.0) - pt) ** self.gamma * loss_bce
        return loss_f.mean()
    
# define loss function using in training
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    
    def forward(self,y_pred, y_true):
        
        #y_pred = y_pred.float()
        #y_true = y_true.float()
        #diceBCELoss = DiceBCELoss()
        
        return  0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)