#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:30:13 2022

@author: chuhsuanlin
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def confusion_matrix(y_true, y_pred):
    
    con_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = con_matrix)
    cm_display.plot()
    plt.show()
    
    return con_matrix
    
def roc_curve(y_true, y_pred):
    
    fpr, tpr, thresholds  = metrics.roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
def quadratic_kappa(y_true, y_pred, N=5):

    w = np.zeros((N,N))
    O = metrics.confusion_matrix(y_true, y_pred)
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in y_true: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in y_pred: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist);
    E = E/E.sum();
    O = O/O.sum();
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
            
    return (1 - (num/den))

def statistical_score(y_true, y_pred):   
    """
    cal all metircs for multi-label classification

    Parameters
    ----------
    y_true : array
        ground truth, label.
    y_pred : array
        predicted value.

    Returns
    -------
    None.

    """
    con_matrix = confusion_matrix(y_true, y_pred)
    
    FP = con_matrix.sum(axis=0) - np.diag(con_matrix) 
    FN = con_matrix.sum(axis=1) - np.diag(con_matrix)
    TP = np.diag(con_matrix)
    TN = con_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return [TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC]
    

def main():
   
    actual = np.asarray([4, 4, 3, 4, 4, 4, 1, 1, 2, 1]) #np.random.binomial(1,.9,size = 1000)
    predicted = np.asarray([0, 2, 1, 0, 0, 0, 1, 1, 2, 1]) #np.random.binomial(1,.9,size = 1000)
    #statistical_score(actual, predicted)
    qua_kappa = quadratic_kappa(actual, predicted)

if __name__ == "__main__":
    main()


