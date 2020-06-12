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
import settings
from utils import print_msg


def batch_get_adjustment_weights(batch_corr_mat, weights_list, adjustment_strategy):
    """
    correct: 1 incorrect: -1
    return corr_w, incorr_w
    TODO: some bugs here. Don't know why.
    """
    # if adjustment_strategy == 1:
    #     pass
    # elif adjustment_strategy == 2:
    #     batch_corr_mat[batch_corr_mat==-1] = 0
    # elif adjustment_strategy == 3:
    #     batch_corr_mat[batch_corr_mat==1] = 0

    # corr_mat = np.sum(batch_corr_mat, axis=0)
    # sign_mat = np.sign(corr_mat)
    # abs_mat = np.abs(corr_mat)
    corr_w_list = []
    incorr_w_list = []

    for i in range(batch_corr_mat.shape[0]):
        corr_mat = batch_corr_mat[i, :]
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
            else: # lite version NOT used
                if len(corr_sets) != 0:
                    corr_id = random.randint(0, len(corr_sets) - 1)
                    corr_w = corr_weights[corr_id]
                if len(incorr_sets) != 0:
                    incorr_id = random.randint(0, len(incorr_sets) - 1)
                    incorr_w = incorr_weights[incorr_id]
        except:
            corr_w = None
            incorr_w = None

        if corr_w is None:
            print_msg('curr w is none.')
        if incorr_w is None:
            print_msg('incorr w is none.')
        

        corr_w_list.append(corr_w)
        incorr_w_list.append(incorr_w)
    
    return corr_w_list, incorr_w_list


def batch_adjust_weights_func(curr_weights, corr_w_list, incorr_w_list, adjustment_strategy, activation='binary'):
    """
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
    adjust_weights = curr_weights
    for corr_w, incorr_w in zip(corr_w_list, incorr_w_list):
        print_msg('here', flush=True)
        if adjustment_strategy == 1:
            if corr_w is None or incorr_w is None:
                print_msg('skip', flush=True)
            else:
                adjust_weights = [item[0] - settings.learning_rate * (item[0] - item[1]) + settings.learning_rate * (item[0] - item[2]) for item in zip(adjust_weights, corr_w, incorr_w)]


        if adjustment_strategy == 2:
            if corr_w is None:
                continue
            else:
                adjust_weights = [item[0] - settings.learning_rate * (item[0] - item[1]) for item in zip(curr_weights, corr_w)]
        if adjustment_strategy == 3:
            if incorr_w is None:
                continue
            else:
                adjust_weights = [item[0] + settings.learning_rate * (item[0] - item[1]) for item in zip(curr_weights, incorr_w)]
                
        if adjustment_strategy == 4:
            if corr_w is None or incorr_w is None:
                continue
            else:
                diff_corr_w = get_difference_func(curr_weights, corr_w, activation=activation)
                diff_incorr_w = get_difference_func(curr_weights, incorr_w, activation=activation)
                adjust_weights = [item[0] - settings.learning_rate * np.multiply(item[0], item[1]) + settings.learning_rate * np.multiply(item[0], item[2]) for item in zip(curr_weights, diff_corr_w, diff_incorr_w)]            
        
        if adjustment_strategy == 5:
            if corr_w is None:
                continue
            else:
                diff_corr_w = get_difference_func(curr_weights, corr_w, activation=activation)
                adjust_weights = [item[0] - settings.learning_rate * np.multiply(item[0], item[1]) for item in zip(curr_weights, diff_corr_w)]
        
        if adjustment_strategy == 6:
            if incorr_w is None:
                continue
            else:
                diff_incorr_w = get_difference_func(curr_weights, incorr_w, activation=activation)
                adjust_weights = [item[0] + settings.learning_rate * np.multiply(item[0], item[1]) for item in zip(curr_weights, diff_incorr_w)]

        # curr_weights = adjust_weights
    return adjust_weights