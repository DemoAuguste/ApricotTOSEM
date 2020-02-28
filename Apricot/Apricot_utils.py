"""
Functions for implementing Apricot+ and Apricot+ Lite.
---------
Notes
---------
Be attention to the order of submodels and weights list.

-------------------
adjustment strategy
-------------------
1: calculate the means of corr_set and incorr_set
2: calculate the means of corr_set
3: calculate the means of incorr_set
4: randomly choose one from corr_set and one from incorr_set
5: randomly choose one from corr_set
6: randomly choose one from incorr_set

"""
import numpy as np
import copy
import os
from keras.layers import Input
from model import *


def cal_avg(weights_list):
    """
    ----------------------
    calculate the average weights of a weights list.
    ----------------------
    return:
    average weights of weights list. format: equals to model.get_weights()
    """
    sum_w = None
    total_num = len(weights_list)
    def weights_add(sum_w, w):
        if sum_w is None:
            sum_w = copy.deepcopy(w)
        else:
            sum_w = [sum(i) for i in zip(sum_w, w)]
        return sum_w
    
    for w in weights_list:
        sum_w = weights_add(sum_w, w)
    sum_w = [item / total_num for item in sum_w]
    
    return sum_w


def get_difference_func(w1, w2, activation='binary'):
    """
    for adjustment strategies 4-6.
    activation: binary, sigmoid, relu, identity
    """
    # Binary: -1, 0 , 1
    def binary(x):
        x[x>0] = 1
        x[x<0] = -1
        return x
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) - 0.5
    
    def relu(x):
        return np.maximum(0, x)
    
    ret_w = None
    
    if activation == 'binary':
        ret_w = [binary(item[0] - item[1]) for item in zip(w1, w2)]
    elif activation == 'sigmoid':
        ret_w = [sigmoid(item[0] - item[1]) for item in zip(w1, w2)]
    elif activation == 'relu':
        ret_w = [relu(item[0] - item[1]) for item in zip(w1, w2)]
    elif activation == 'identity':
        ret_w = [item[0] - item[1] for item in zip(w1, w2)]
    
    return ret_w


def get_adjustment_weights(corr_mat, weights_list, adjustment_strategy):
    """
    return corr_w, incorr_w
    TODO: some bugs here. Don't know why.
    """
    try:
        corr_sets = np.nonzero(corr_mat)[0].tolist()
        if corr_sets is None or  len(corr_sets) == 0:
            corr_sets = []
            corr_weights = []
        else:
            corr_weights = [weights_list[i] for i in corr_sets]
        temp_incorr_matrix = np.ones(shape=corr_mat.shape) - corr_mat
        incorr_sets = np.nonzero(temp_incorr_matrix)[0].tolist()
        if incorr_sets is None or len(incorr_sets) == 0:
            incorr_sets = []
            incorr_weights = []
        else:
            incorr_weights = [weights_list[i] for i in incorr_sets]
        
        corr_w = None
        incorr_w = None
        
        if adjustment_strategy == 1 or adjustment_strategy == 2 or adjustment_strategy == 3:
            if len(corr_sets) != 0:
                corr_w = cal_avg(corr_weights)
            if len(incorr_sets) != 0:
                incorr_w = cal_avg(incorr_weights)    
        else: # lite version
            if len(corr_sets) != 0:
                corr_id = random.randint(0, len(corr_sets) - 1)
                corr_w = corr_weights[corr_id]
            if len(incorr_sets) != 0:
                incorr_id = random.randint(0, len(incorr_sets) - 1)
                incorr_w = incorr_weights[incorr_id]
    except:
        corr_w = None
        incorr_w = None
    
    return corr_w, incorr_w

