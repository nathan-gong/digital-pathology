# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import pandas as pd
import os

df = pd.read_csv('/work/digital-pathology/dataset/train.csv')
img_names = df['image_id']
PATH = '/work/digital-pathology/dataset/tile-images-64-196/'

for name in img_names:
    name = name +'_0.jpg'
    #print(name)
    img_path = os.path.join(PATH, name)
    
    if os.path.exists(img_path):
        pass
    else:
        print(img_path)
        

