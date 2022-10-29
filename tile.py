#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:48 2022

@author: chuhsuanlin
"""


import os
import cv2
#import PIL
#import random

import matplotlib
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, display
from PIL import Image
import openslide
from glob import glob
import sys
import argparse


def tile(tile_size,tile_num,img_name,SAVE_DIR, plot=False):
    
    
    s = np.sqrt(tile_num)

    if s-int(s) !=0:
        sys.exit(f'tile number must be square number') 
    
    else:
        s = int(s)
        
    img_path = os.path.join(PATH, "train_images/", img_name)
    img = openslide.OpenSlide(img_path)
    img_h,img_w = img.dimensions
    
    sum_array = []
    for i in range(0,img_h//tile_size[1]):
        r = i*tile_size[1]
        for j in range(0,img_w//tile_size[1]):
            c = j*tile_size[1]
            
            #print(r,c)
            tile_img = img.read_region(((r,c)), 0, tile_size)
            tile = np. array(tile_img)
            tile_sum = tile.sum()
            sum_array.append(tile_sum)
            
    
    selected_img = np.argsort(sum_array)
    if plot:
        fig = plt.figure()
    
    
    TILE_DIR = os.path.join(SAVE_DIR, img_name[:-5])
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
     
    tiles=[]
    for i in range(tile_num):
        r = selected_img[i]//(j+1) * tile_size[0]
        c = selected_img[i]%(j+1)* tile_size[0]
        
        img_patch = np.array(img.read_region(((r,c)), 0, tile_size))
        
        tile_path = os.path.join(SAVE_DIR,f'{img_name[:-5]}_{i}.jpg')
        cv2.imwrite(tile_path, img_patch)
        
         
        if plot:
            ax = fig.add_subplot(s, s, i+1)  
            ax.imshow(img_patch)
            plt.axis('off')  
        
    img.close()
    
    
        
                    
                    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, default = 256)
    parser.add_argument('-num', type=int, default = 16)
    parser.add_argument('-path', type=str, default = '/work/digital-pathology/dataset/')
    args = parser.parse_args()       
    
    tile_size = (args.size, args.size)
    tile_num = args.num
    global PATH
    PATH = args.path
    
    all_train_images = os.listdir(os.path.join(PATH,"train_images/"))
    train_df = pd.read_csv(os.path.join(PATH,'train.csv'))
    SAVE_DIR = os.path.join(PATH,f"tile_images_{tile_num}_{args.size}/")
    is_Exist = os.path.exists(SAVE_DIR)
    if not is_Exist:
        os.makedirs(SAVE_DIR)
    
    
    s = np.sqrt(tile_num)

    if s-int(s) !=0:
        sys.exit(f'tile number must be square number') 
    
    for img_name in all_train_images:
        if img_name[-4:] == "tiff":
                    
            tile(tile_size,tile_num,img_name,SAVE_DIR)

       
    
if __name__ == "__main__":
    main()

    