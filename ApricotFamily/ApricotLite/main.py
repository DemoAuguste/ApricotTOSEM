from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils import load_dataset, split_validation_dataset
from utils import logger
from settings import NUM_SUBMODELS, LOOP_COUNT, FIX_BATCH_SIZE
from .func import *
from settings import BATCH_SIZE, FURTHER_ADJUSTMENT_EPOCHS
import os
import numpy as np
from ApricotFamily.ApricotPlus.func import batch_get_adjust_w
from ApricotFamily.Apricot.func import get_indexed_failing_cases, apricot_cal_sub_corr_mat, get_weights_list
from datetime import datetime


def apricot_lite(model, model_weights_dir, dataset, adjustment_strategy=None):
    """
    apricot lite
    does not need strategy.
    """
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    train_size = len(x_train)
    val_size = len(x_train_val)
    test_size = len(x_test)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'apricot_lite_fixed.h5')
    log_path = os.path.join(model_weights_dir, 'apricot_lite.log')

    if not os.path.join(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    fixed_model.load_weights(trained_weights_path)

    start = datetime.now()
    logger('---------------original model---------------', log_path)
    _, base_train_acc = fixed_model.evaluate(x_train_val, y_train_val)
    _, base_val_acc = fixed_model.evaluate(x_val, y_val)
    _, base_test_acc = fixed_model.evaluate(x_test, y_test)

    logger('train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(base_train_acc, base_val_acc, base_test_acc),
           log_path)

    fail_xs, fail_ys, fail_ys_label, fail_num, fail_index = get_indexed_failing_cases(fixed_model, x_train, y_train)
    print('getting sub correct matrix...')
    sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_mat_{}.npy'.format(NUM_SUBMODELS))

    if not os.path.exists(sub_correct_matrix_path):
        print('generating matrix....')
        sub_correct_mat = apricot_cal_sub_corr_mat(fixed_model, submodel_dir, fail_xs, fail_ys, num_submodels=NUM_SUBMODELS)
        np.save(sub_correct_matrix_path, sub_correct_mat)
    else:
        print('loading matrix...')
        sub_correct_mat = np.load(sub_correct_matrix_path)

    fixed_model.load_weights(trained_weights_path)

    weights_list = get_weights_list(fixed_model, submodel_dir, NUM_SUBMODELS)

    best_train_acc = base_train_acc
    best_val_acc = base_val_acc
    best_test_acc = base_test_acc

    # Apricot Plus: iterates failing cases.
    print('start the main iteration process...')

    # Apricot Plus: iterates failing cases.
    print('start the main iteration process...')
    for count in range(LOOP_COUNT):  # iterate 3 times.
        np.random.shuffle(sub_correct_mat)
        # for i in range()
        iter_count, res = divmod(sub_correct_mat.shape[0], FIX_BATCH_SIZE)
        if res != 0:
            iter_count += 1
        for i in range(iter_count):
            curr_w = fixed_model.get_weights()
            batch_corr_mat = sub_correct_mat[FIX_BATCH_SIZE * i: FIX_BATCH_SIZE * (i + 1)]  # 20 samples in one batch
            adjust_w = batch_lite_get_adjust_w(curr_w, batch_corr_mat, weights_list)

            fixed_model.set_weights(adjust_w)

            x = int(count * sub_correct_mat.shape[0] + i + 1)
            y = int(LOOP_COUNT * sub_correct_mat.shape[0])
            _, curr_acc = fixed_model.evaluate(x_val, y_val)
            print('[iteration {}/{}] current val acc: {:.4f}'.format(x, y, curr_acc))

            if curr_acc > best_val_acc:
                best_val_acc = curr_acc
                fixed_model.save_weights(fixed_weights_path)
                logger('Improved. val acc: {:.4f}'.format(best_val_acc), log_path)
            else:
                fixed_model.load_weights(fixed_weights_path)

    end = datetime.now()
    logger('Spend time: {}'.format(end - start), log_path)











