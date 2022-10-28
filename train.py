#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:29:19 2022

@author: chuhsuanlin
"""

import os
import sys
import pandas as pd
from glob import glob
import gc
import numpy as np
import matplotlib.pyplot as plt
import argparse


import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary


from sklearn.model_selection import StratifiedGroupKFold
from trainer import get_loader, get_model, get_loss
from general import read_yaml
import segmentation_models_pytorch as smp
from pandadataset import PANDADataset
#import torch.optim.lr_scheduler.StepLR

PATH = './Data/' #args.path

'''
train_images=[]
all_train_images = os.listdir(os.path.join(PATH,"train_images/"))
for img_name in all_train_images:
    if img_name[-4:] == "tiff":
        train_images.append(img_name[:-5]) 
'''       
cfg = read_yaml()     
k_fold = cfg.Data.dataset.kfold   
tiles = os.listdir(os.path.join(PATH,"tile_images/"))
df = pd.read_csv(os.path.join(PATH,f'train-{k_fold}fold.csv'))

'''
train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

df_test = train_df.loc[train_df['image_id'].isin(train_images)]
df_test = df_test.reset_index(drop=True)
df_test.to_csv('../Data/new_train.csv')
'''

model = get_model(cfg)
#summary(model, (3, 256, 256))
# check if gpu exist 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Using GPU or CPU to train
model.to(device)  




# zero the parameter gradients
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
log_interval = 5
epoch = cfg.Data.dataloader.epoch

for k in range(1,k_fold):
    train_df = df.loc[df['kfold']==k].reset_index(drop=True)
    valid_df = df.loc[df['kfold']!=k].reset_index(drop=True)
    
    train_loader = get_loader(train_df, "train", cfg)
    valid_loader = get_loader(valid_df, "valid", cfg)
    
    lossfunction = get_loss(cfg)
    
    for e in range(0,epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            mask, logits = model(images)
            
            loss = lossfunction(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            
            if batch_idx % log_interval == 0:              
                print(f'Train Epoch: {epoch} [{(batch_idx+1) * len(images)}/{len(train_loader.dataset)} ({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
             
            
            
            model.eval()
            test_loss = 0;
            
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    mask, logits = model(images)
                    test_loss += lossfunction(logits, labels)
                    
                    pred = logits.argmax()
                    
                    
        
    
    
    
    






