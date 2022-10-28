#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:17:24 2022

@author: chuhsuanlin

https://www.kaggle.com/code/gpreda/panda-challenge-starting-eda/notebook

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
import seaborn as sns

PATH = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/'
train_df = pd.read_csv(os.path.join(PATH,'train.csv'))
test_df = pd.read_csv(os.path.join(PATH,'test.csv'))

#print(train_df.head())

train_image_list = os.listdir(os.path.join(PATH, 'train_images'))
train_label_masks_list = os.listdir(os.path.join(PATH, 'train_label_masks'))


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index)
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
    

plot_count(train_df, 'data_provider')
plot_count(train_df, 'isup_grade')
plot_count(train_df, 'gleason_score',size =3)        

fig, ax = plt.subplots(nrows=1,figsize=(12,6))
tmp = train_df.groupby('isup_grade')['gleason_score'].value_counts()
df = pd.DataFrame(data={'Exams': tmp.values}, index=tmp.index).reset_index()
sns.barplot(ax=ax,x = 'isup_grade', y='Exams',hue='gleason_score',data=df, palette='Set1')
plt.title("Number of examinations grouped on ISUP grade and Gleason score")
plt.show()


# show image & mask
file_name = '0005f7aaab2800f6170c399693a96917'

img_folder = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/train_images/'
mask_folder = '/Users/chuhsuanlin/Documents/NEU/Course/Fall 2022/BIOE 5860 Precision Medicine/Data/train_label_masks/'
img_path = os.path.join(img_folder,file_name+'.tiff')
mask_path = os.path.join(mask_folder,file_name+'_mask.tiff')

img = openslide.OpenSlide(img_path)
mask = openslide.OpenSlide(mask_path)

idx = train_df[train_df["image_id"] == file_name].index.to_numpy()

if train_df['data_provider'][idx].values == 'karolinska':
    
    cmap = matplotlib.colors.ListedColormap(['white', 'gray', 'green', 'yellow', 'orange', 'red'])
else:
    cmap = matplotlib.colors.ListedColormap(['white', 'green', 'red'])
    
img_data = img.read_region((0,0), img.level_count-1, img.level_dimensions[-1])
mask_data = mask.read_region((0,0), mask.level_count-1, mask.level_dimensions[-1])

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)  
ax.imshow(img_data) 
ax = fig.add_subplot(2, 1, 2)  
ax.imshow(np.asarray(mask_data)[:,:,0],cmap=cmap)






