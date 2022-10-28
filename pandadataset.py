#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:58:46 2022

@author: chuhsuanlin
"""


import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from torch.utils.data import Dataset, DataLoader
import torch
import os


class PANDADataset(Dataset):
    def __init__(self, df, cfg_image, phase, transform):

        assert phase in {"train", "valid"}
        
        self.df = df
        self.transform = transform
        self.cfg_imgae = cfg_image
        self.img_dir = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/tile_images/'
        self.num_tile = 16
        self.img_size = 256
        self.phase = phase
        self.labels = self.df["isup_grade"]
        
        '''
        # Tile augmentation
        if phase == "train":
            
            self.random_tile = conf_dataset.random_tile
            self.random_rotate_tile = conf_dataset.rotate_tile
            self.random_tile_white = conf_dataset.random_tile_white
            
        elif phase == "valid":
            self.random_tile = False
            self.random_rotate_tile = False
            self.random_tile_white = False
        '''
    
    def get_normal_tile(self, idx):
        file_name = self.df["image_id"].values[idx]
        paths = [
            os.path.join(self.img_dir, f"{file_name}_{idx}.jpg") for idx in range(self.num_tile)
        ]
        return self.get_tile_from_paths(paths)
    
    
    def get_tile_from_paths(self, paths):
        # BGR
        tiles = np.array([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths])
        return tiles
      
    
    def tile_concat(self, tiles):
        
        s = int(np.sqrt(len(tiles)))
        
        for i, img_patch in enumerate(tiles):
            
            if i%s == 0:            
                h_img = img_patch
                       
            else:
                h_img = cv2.hconcat([h_img, img_patch])
                
                if i%s == (s-1) :
                    if i//s ==0:
                        images = h_img
                    else:
                        images = cv2.vconcat([images, h_img])
                        
        return images
    
    def get_image(self, index):
        
        tiles =self.get_normal_tile(index)
        
        if self.phase == 'train':
            pass
            #aug
            #tiles = tiles_aug;            
         
        return self.tile_concat(tiles)
        
           
    def get_labels(self, index):
        
        label = np.zeros(6).astype(np.float32)    
        label[self.labels[index]] = 1.0
        
        #self.df.head()
        return label
            
    
    def img_preprocess(self, img):
        # resize and normalization
        # TODO same img preprocess in kernel.py
        img = img.astype(np.float32)
        img = 255 - img  # this task imgs has many whites(255)
        img /= 255  # ToTensorV2 has no normalize !
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img
        
    def __getitem__(self, index): 

        print(index)
        # get tiles
        img = self.get_image(index)
        img = self.img_preprocess(img)
        
        if self.transform:
            data = self.transform(image=img)
        image = data['image']
        
        label = self.get_labels(index)
        
        return image, label
    
        
    def __len__(self):

        return len(self.df)
    
    
    
    

    
    
    
    
    
    