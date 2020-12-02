from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils import load_dataset, split_validation_dataset
from utils import logger
from settings import NUM_SUBMODELS, LOOP_COUNT, FIX_BATCH_SIZE
from .func import *
from settings import BATCH_SIZE, FURTHER_ADJUSTMENT_EPOCHS
import os
import numpy as np
from .func import *
from ApricotFamily.Apricot.func import get_indexed_failing_cases, apricot_cal_sub_corr_mat, get_weights_list


def apricot_plus(model, model_weights_dir, dataset, adjustment_strategy):
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    train_size = len(x_train)
    val_size = len(x_train_val)
    test_size = len(x_test)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'apricot_plus_fixed_{}.h5'.format(adjustment_strategy))
    log_path = os.path.join(model_weights_dir, 'apricot_plus_log_{}.h5')

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

    # to simply the process, get the classification results of submodels first.
    # do not shuffle the training dataset.
    fail_xs, fail_ys, fail_ys_label, fail_num, fail_index = get_indexed_failing_cases(fixed_model, x_train, y_train)

    sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_mat_{}.npy'.format(NUM_SUBMODELS))
    if not os.path.exists(sub_correct_matrix_path):
        sub_correct_mat = apricot_cal_sub_corr_mat(fixed_model, submodel_dir, fail_xs, fail_ys, num_submodels=NUM_SUBMODELS)
        np.save(sub_correct_matrix_path, sub_correct_mat)
    else:
        sub_correct_mat = np.load(sub_correct_matrix_path)

    fixed_model.load_weights(trained_weights_path)

    # Apricot Plus: iterates failing cases.
    sub_weights_list = get_weights_list(fixed_model, submodel_dir, num_submodels=NUM_SUBMODELS)
    for _ in range(LOOP_COUNT):  # iterate 3 times.
        np.random.shuffle(sub_correct_mat)
        for i in range(sub_correct_mat.shape[0]):
            curr_w = fixed_model.get_weights()
            batch_corr_mat = sub_correct_mat[FIX_BATCH_SIZE*i: FIX_BATCH_SIZE*(i+1)]  # 20 samples in one batch
            corr_w, incorr_w = batch_get_adjustment_weights(batch_corr_mat, sub_weights_list, adjustment_strategy,
                                                            curr_w)


