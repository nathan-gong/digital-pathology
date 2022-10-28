#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:38:41 2022

@author: chuhsuanlin
"""


import yaml
from addict import Dict

def read_yaml(fpath="./configs/sample.yaml"):
  
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
     

if __name__ == "__main__":
        
   yml = read_yaml()
    
    
    
    