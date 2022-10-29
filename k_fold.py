#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:02:19 2022

@author: chuhsuanlin
"""
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse

def k_fold(train_df, n_splits, path):


    # k cross- validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = np.zeros(len(train_df))
    labels = train_df['isup_grade']
    for i, (train_index, val_index) in enumerate(kfold.split(X, labels)):
            train_df.loc[val_index, "kfold"] = i + 1
     
    #out_path = 
    train_df.to_csv(os.path.join(path,f'train-{n_splits}fold.csv'))


   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default = '/work/digital-pathology/dataset/')
    parser.add_argument('-k', type=int, default = 5)
    parser.add_argument('-file', type=str, default ='train')
    
    args = parser.parse_args()     
    
    path = args.path
    n_splits = args.k
    file = args.file
    
    train_df = pd.read_csv(os.path.join(path,f'{file}.csv'))

    k_fold(train_df,n_splits, path)