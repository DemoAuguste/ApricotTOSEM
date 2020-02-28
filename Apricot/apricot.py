"""
main function of apricot+ and apricot+ lite revision.
"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input
from datetime import datetime
import os
from utils.utils import logger
import numpy as np
from utils import *
from .Apricot_utils import *


def cal_sub_corr_matrix(model, corr_path, submodels_path, fail_xs, fail_ys_label, fail_num):
    sub_correct_matrix = None

    for root, dirs, files in os.walk(submodels_path):
        for f in files:
            temp_w_path = os.path.join(root, f)
            model.load_weights(temp_w_path)
            sub_y_pred = model.predict(fail_xs)

            sub_col = np.argmax(sub_y_pred, axis=1) - fail_ys_label
            sub_col[sub_col != 0] = 1

        if sub_correct_matrix is None:
            sub_correct_matrix = sub_col.reshape(fail_num, 1)
        else:
            sub_correct_matrix = np.concatenate((sub_correct_matrix, sub_col.reshape(fail_num, 1)), axis=1)

        sub_correct_matrix = np.ones(shape=sub_correct_matrix.shape) - sub_correct_matrix  # here change 0 to 1 (for correctly predicted case)
        np.save(corr_path, sub_correct_matrix)

    return sub_correct_matrix

            

def get_failing_cases(model, xs, ys):
    y_preds = model.predict(xs)
    y_pred_label = np.argmax(y_preds, axis=1)
    y_true = np.argmax(ys, axis=1)

    index_diff = np.nonzero(y_pred_label - y_true)

    fail_xs = xs[index_diff]
    fail_ys = ys[index_diff]
    fail_ys_label = np.argmax(fail_ys, axis=1)
    fail_num = int(np.size(index_diff))

    return fail_xs, fail_ys, fail_ys_label, fail_num


def apricot(model, model_weights_dir, dataset, adjustment_strategy):
    """
    input:
        * dataset: [x_train_val, y_train_val, x_val, y_val, x_test, y_test]
    """
    # package the dataset
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    fixed_model = model

    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')  
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed.h5')
    log_path = os.path.join(model_weights_dir, 'log.txt')

    if not os.path.exists(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)

    logger('----------original model----------', log_path)

    # submodels 
    _, base_val_acc = fixed_model.evaluate(x_train_val, y_train_val)
    print('The validation accuracy: {:.4f}'.format(base_val_acc))
    _, base_test_acc = fixed_model.evaluate(x_test, y_test)
    print('The test accuracy: {:.4f}'.format(base_test_acc))
    
    best_weights = fixed_model.get_weights()
    best_acc = base_val_acc

    # find all indices of xs that original model fails on them.
    fail_xs, fail_ys, fail_ys_label, fail_num = get_failing_cases(fixed_model, x_train_val, y_train_val)

    sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_matrix_{}.npy'.format(random_seed))
    sub_correct_matrix = None # 1: predicts correctly, 0: predicts incorrectly.

    if not os.path.exists(sub_correct_matrix_path):
        # obtain submodel correctness matrix
        sub_correct_matrix = cal_sub_corr_matrix(fixed_model, sub_correct_matrix_path, submodel_dir, fail_xs, fail_ys_label, fail_num)
    else:
        sub_correct_matrix = np.load(sub_correct_matrix_path)

    sub_weights_list = get_submodels_weights(fixed_model, submodel_dir)

    fixed_model.load_weights()

    for _ in range(settings.LOOP_COUNT):
        np.random.shuffle(sub_correct_matrix)

        for index in range(sub_correct_matrix.shape[0]):
            curr_weights = fixed_model.get_weights()
            corr_mat = sub_correct_matrix[index, :]
            
            corr_w, incorr_w = get_adjustment_weights(corr_mat, sub_weights_list, adjustment_strategy)
            adjust_w = adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
            

    
    


def apricot_lite():
    pass