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


def get_submodels_weights(model, model_name, dataset, path):
    """
    model: the model to be loaded.
    path: the dir path to the submodel weights.

    return: weights list
    """
    if dataset == 'cifar10':
        img_rows, img_cols = 32, 32
        img_channels = 3
        num_classes = 10
        top_k = 1
    elif dataset == 'cifar100':
        img_rows, img_cols = 32, 32
        img_channels = 3
        num_classes = 100
        top_k = 5
    else:
        pass
    
    input_tensor = Input(shape=(img_rows, img_cols, img_channels))

    if model_name == 'resnet20':
        fixed_model = build_resnet(img_rows, img_cols, img_channels, num_classes=num_classes, stack_n=3, k=top_k)
    elif model_name == 'resnet32':
        fixed_model = build_resnet(img_rows, img_cols, img_channels, num_classes=num_classes, stack_n=5, k=top_k)
    elif model_name == 'mobilenet':
        fixed_model = build_mobilenet(input_tensor, num_classses=num_classes, k=top_k)
    elif model_name == 'mobilenet_v2':
        fixed_model = build_mobilenet_v2(input_tensor, num_classses=num_classes, k=top_k)
    elif model_name == 'densenet':
        fixed_model = build_densenet(input_tensor, num_classses=num_classes, k=top_k)

    weights_list = []
    for root, dirs, files in os.walk(path):  # os.path.join(original_dir_path, 'submodels')
        for name in files:
            file_path = os.path.join(root, name)
            fixed_model.load_weights(file_path)
            weights_list.append(fixed_model.get_weights())
    return weights_list