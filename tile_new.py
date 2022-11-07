#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:00:30 2022

@author: chuhsuanlin
"""

import logging
from datetime import datetime
import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np

TRAIN = '/work/digital-pathology/dataset/train_images/'
MASKS = '/work/digital-pathology/dataset/train_label_masks/'
OUT_TRAIN = '/work/digital-pathology/dataset/train.zip'
OUT_MASKS = '/work/digital-pathology/dataset/masks.zip'
sz = 256
N = 16

# set up logging
logging.basicConfig(
    filename='log.txt',
    filemode='w',
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt=datetime.utcnow().astimezone().replace(
        microsecond=0).isoformat(),
    level=logging.DEBUG)

PATH = 'work/digital-pathology/dataset/'
SAVE_DIR = os.path.join(PATH,f"tile-images-{N}-{sz}/")
is_Exist = os.path.exists(SAVE_DIR)
if not is_Exist:
    os.makedirs(SAVE_DIR)

def tile(img, mask):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


x_tot,x2_tot = [],[]
names = [name[:-10] for name in os.listdir(MASKS)]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
 zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for name in (names):
        logging.debug(name)
        img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]
        mask = skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[-1]
        tiles = tile(img,mask)
        for t in tiles:
            img,mask,idx = t['img'],t['mask'],t['idx']
            x_tot.append((img/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0)) 
            #if read with PIL RGB turns into BGR
            img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}_{idx}.png', img)
            mask = cv2.imencode('.png',mask[:,:,0])[1]
            mask_out.writestr(f'{name}_{idx}.png', mask)
            
