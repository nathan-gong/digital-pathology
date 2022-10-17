#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:49:45 2022

@author: chuhsuanlin
"""

import os
#import cv2
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


tile_size = (1024,1024)
tile_num = 20;
file_name = '0005f7aaab2800f6170c399693a96917'

img_folder = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/train_images/'
mask_folder = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/train_label_masks/'
img_path = os.path.join(img_folder,file_name+'.tiff')
mask_path = os.path.join(mask_folder,file_name+'_mask.tiff')

img = openslide.OpenSlide(img_path)
mask = openslide.OpenSlide(mask_path)

#small = mask.get_thumbnail(size=(256,256))
#small.show()

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
fig = plt.figure()
for i in range(tile_num):
    r = selected_img[i]//(j+1) * tile_size[0]
    c = selected_img[i]%(j+1)* tile_size[0]
    
    #print(r,c)
    img_patch = img.read_region(((r,c)), 0, tile_size)
    mask_patch = mask.read_region(((r,c)), 0, tile_size)
    
    
    ax = fig.add_subplot(4, 5, i+1)  
    ax.imshow(img_patch)
    #ax.imshow(np.asarray(mask_patch)[:,:,0],cmap=cmap)
    plt.axis('off')  
    
img.close()











