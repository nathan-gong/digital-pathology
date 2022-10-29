#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:29:19 2022

@author: chuhsuanlin
"""

import os
#import sys
import pandas as pd
#import argparse


import torch
import torch.optim as optim
from torchsummary import summary


from trainer import get_loader, get_model, get_loss
from general import read_yaml


PATH = '/work/digital-pathology/dataset/' #args.path
yaml_name = 'sample.yaml'  

'''
train_images=[]
all_train_images = os.listdir(os.path.join(PATH,"train_images/"))
for img_name in all_train_images:
    if img_name[-4:] == "tiff":
        train_images.append(img_name[:-5]) 
'''     

cfg = read_yaml(yaml_name)     
k_fold = cfg.Data.dataset.kfold   
tile_size = cfg.Data.dataset.tile_size
tile_num = cfg.Data.dataset.num_tile
    
tiles = os.listdir(os.path.join(PATH,f"tile-images-{tile_num}-{tile_size}/"))
df = pd.read_csv(cfg.Data.dataset.train_df)

'''
train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

df_test = train_df.loc[train_df['image_id'].isin(train_images)]
df_test = df_test.reset_index(drop=True)
df_test.to_csv('../Data/new_train.csv')
'''

#model = get_model(cfg)
#summary(model, (3, 256, 256))
# check if gpu exist 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Using GPU or CPU to train
#model.to(device)  

output_path = './output'
is_Exist = os.path.exists(output_path)
if not is_Exist:
    os.makedirs(output_path)
        
model_name = cfg.Model.base

log_interval = 5
epoch = cfg.Data.dataloader.epoch

for k in range(1,k_fold):
    
    print(f' ==== {k} fold ====')
    # model setting
    model = get_model(cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Using GPU or CPU to train
    model.to(device)

    # zero the parameter gradients
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # load data
    train_df = df.loc[df['kfold']==k].reset_index(drop=True)
    valid_df = df.loc[df['kfold']!=k].reset_index(drop=True)
    
    train_loader = get_loader(train_df, "train", cfg)
    valid_loader = get_loader(valid_df, "valid", cfg)
    
    # define loss function
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
            valid_loss = 0
            correct = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    mask, logits = model(images)
                    valid_loss += lossfunction(logits, labels)
                    
                    pred = logits.argmax()
                    label = labels.argmax()
                    
                    correct += torch.eq(pred, label).sum().item()
                    
                accuracy = correct/len(valid_loader.dataset)*100
                valid_loss/=len(valid_loader.dataset)
                print(f'\nValidation set: Average loss = {valid_loss}, Accuracy = {accuracy}%')
                
            scheduler.step()
        
         
        
        model_path = os.path.join(output_path,f'{yaml_name}_{model_name}_{k}.pth')
        torch.save(model.state_dict(), model_path)
                
                
            
                    
                    
        
    
    
    
    






