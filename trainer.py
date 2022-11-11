#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:53:27 2022

@author: chuhsuanlin
"""

from pandadataset import PANDADataset
import albumentations as A
import albumentations.pytorch as Albp

from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
import segmentation_models_pytorch as smp
from loss import FocalLoss
from efficientnet import CustomEfficientNet,enetv2
#from transformers import BeitFeatureExtractor, BeitForImageClassification
#from transformers import BeitConfig


def transform(phase, cfg):
    assert phase in {"train", "valid"}
    
    img_size = cfg.Data.dataset.img_size
    if phase == "train":
        transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        Albp.ToTensorV2()
        
        ])
        
    else:       
        transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        Albp.ToTensorV2()
        ])
        
    return transform
            
    
    
def get_loader(df, phase, cfg):
    assert phase in {"train", "valid"}
    
    transforms = transform(phase, cfg)
    dataset = PANDADataset(
        df = df, 
        cfg = cfg, 
        phase = phase, 
        transform = transforms)
        

    #cfg_dataloader = self.cfg.Data.dataloader
    
    return DataLoader(
        dataset,
        batch_size=cfg.Data.dataloader.batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=cfg.Data.dataloader.num_workers,
        drop_last=True if phase == "train" else False,
       #worker_init_fn=worker_init_fn,
    )


def get_model(cfg):
    
    #Beit_cfg = BeitConfig(image_size = 256, num_labels = 6)
    #net = BeitForImageClassification(Beit_cfg)


    '''
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=6,                 # define number of output labels
    )


    # U-Net with efficientnet-b1
    net = smp.Unet(
    encoder_name="efficientnet-b1",      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #classes=3,        # model output channels (number of classes in your dataset)
    activation=None,
    aux_params = aux_params
    )
    '''
    net = nn.DataParallel(enetv2())
    return net


def get_loss(cfg):
    
    loss = nn.BCEWithLogitsLoss()
    #loss = nn.CrossEntropyLoss()
    return loss
    
    
    





    
    
