# the main process of Apricot (conference version).
import keras
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from utils import load_dataset, split_validation_dataset
from utils import logger
from settings import NUM_SUBMODELS
from .func import *


def apricot(model, model_weights_dir, dataset, adjustment_strategy):
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    train_size = len(x_train)
    val_size = len(x_train_val)
    test_size = len(x_test)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'apricot_fixed_{}.h5'.format(adjustment_strategy))
    log_path = os.path.join(model_weights_dir, 'apricot_log_{}.h5')

    if not os.path.join(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    logger('---------------original model---------------', log_path)
    _, base_train_acc = fixed_model.evaluate(x_train_val, y_train_val)
    _, base_val_acc = fixed_model.evaluate(x_val, y_val)
    _, base_test_acc = fixed_model.evaluate(x_test, y_test)

    best_weights = fixed_model.get_weights()  # used for keeping the best weights of the model.
    best_train_acc = base_train_acc

    # to simply the process, get the classification results of submodels first.
    # do not shuffle the training dataset.
    fail_xs, fail_ys, fail_ys_label, fail_num, fail_index = get_indexed_failing_cases(fixed_model, x_train, y_train)

    sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_mat_{}.npy'.format(NUM_SUBMODELS))
    sub_correct_mat = None
    if not os.path.exists(sub_correct_matrix_path):
        sub_correct_mat = cal_sub_corr_matrix(fixed_model, sub_correct_matrix_path, submodel_dir, fail_xs, fail_ys, fail_ys_label, fail_num, num_submodels=NUM_SUBMODELS)
    else:
        sub_correct_mat = np.load(sub_correct_matrix_path)

    # iterates all training dataset.
    iter_batch_size = 20  # TODO revise hard-coding
    iter_num, ret = divmod(train_size, iter_batch_size)
    fail_idx_seq = get_formatted_batch_sequence(fail_index, total_num=train_size)
    if ret != 0:
        iter_num += 1
    for i in range(iter_num):  # iterates by batch.
        # check if the index is in the fail_index.
        temp_train_index = [i for i in range(iter_batch_size*i, iter_batch_size*(i+1))]
        temp_train_index = np.array(temp_train_index)
        temp_fail_idx_seq = fail_idx_seq[temp_train_index]
        





