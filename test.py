#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:36:22 2022

@author: chuhsuanlin
"""


import os
#import sys
import pandas as pd
#import argparse
import numpy as np

import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from trainer import get_loader, get_model, get_loss
from general import read_yaml

from metrics import confusion_matrix

data_path = '/work/digital-pathology/dataset/' #args.path
model_folder = './output'
test_model = '011.yaml_efficientnet-b2_1_epoch_4.pth'
model_path = os.path.join(model_folder,test_model)

yaml_name = '011.yaml'  
cfg = read_yaml(yaml_name)   

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_model(cfg)
model.load_state_dict(torch.load(model_path,map_location = device))
model.eval()
model.to(device)

test_df = pd.read_csv(cfg.Data.dataset.test_df)
test_loader = get_loader(test_df, "valid", cfg)

label = np.array([])
pred = np.array([])

debug = False
i = 0
with torch.no_grad():
    for images, labels in test_loader:
        i+=1
        images, labels = images.to(device), labels.to(device)
        #print(images)
        logits = model(images)
        #print('image=========')
        #print(images)
        #print('logits=======')
        #print(logits)
        #print('label', labels)
        pred_ = logits.argmax(dim=1).cpu().numpy()
        pred = np.concatenate((pred, pred_), axis = None)
        label_ = labels.argmax(dim=1).cpu().numpy()
        label = np.concatenate((label, label_), axis=None) 
        
        #print('pred=====')
        #print(pred)

        if(debug and i>1):
            break;
        print('evaluating.....')
 
cfx = confusion_matrix(label, pred)
print(cfx)
       
label = label[:,np.newaxis]
pred = pred[:,np.newaxis]
result = np.append(label,pred,axis=1)
#print(label)
#print(pred)
save_path = os.path.join(model_folder, test_model[:-4])
pd.DataFrame(result).to_csv(f'{save_path}.csv')
      
