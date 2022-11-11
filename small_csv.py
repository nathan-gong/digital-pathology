#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:33:01 2022

@author: chuhsuanlin
"""

import os
import numpy as np
import pandas as pd

PATH = '/work/digital-pathology/dataset/'
tile_path = '/work/digital-pathology/dataset/tile-images-16-256/'

names = np.unique([name[:32] for name in os.listdir(tile_path)])

train_df = pd.read_csv(os.path.join(PATH,'train.csv'))

df_small = train_df.loc[train_df['image_id'].isin(names)]
df_small = df_small.reset_index(drop=True)

df_small.to_csv(os.path.join(PATH,'small_train.csv'))
