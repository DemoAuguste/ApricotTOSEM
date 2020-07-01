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
from .Apricot_utils2 import *
import settings
from .Apricot_general_utils import *


def cal_sub_corr_matrix(model, corr_path, submodels_path, fail_xs, fail_ys, fail_ys_label, fail_num, num_submodels=20):
    # add threshold
    sub_correct_matrix = None

    for root, dirs, files in os.walk(submodels_path):
        for i in range(num_submodels):
            temp_w_path = os.path.join(root, 'sub_{}.h5'.format(i))
            model.load_weights(temp_w_path)

            class_prob_mat = get_class_prob_mat(model, fail_xs, fail_ys) # threshold

            sub_col = get_model_correct_mat(model, fail_xs, fail_ys, class_prob_mat)

            # sub_y_pred = model.predict(fail_xs)
            # sub_col = np.argmax(sub_y_pred, axis=1) - fail_ys_label
            # sub_col[sub_col != 0] = 1

            if sub_correct_matrix is None:
                sub_correct_matrix = sub_col.reshape(fail_num, 1)
            else:
                sub_correct_matrix = np.concatenate((sub_correct_matrix, sub_col.reshape(fail_num, 1)), axis=1)
                # print(sub_correct_matrix.shape)

    # original code, no need to change now
    # sub_correct_matrix = np.ones(shape=sub_correct_matrix.shape) - sub_correct_matrix  # here change 0 to 1 (for correctly predicted case)
    
    sub_correct_matrix[sub_correct_matrix == 0] = -1
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


def apricot(model, model_weights_dir, dataset, adjustment_strategy, activation='binary'):
    """
    including Apricot and Apricot lite
    input:
        * dataset: [x_train_val, y_train_val, x_val, y_val, x_test, y_test]
    """
    # package the dataset
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    # x_train_val = np.concatenate((x_train_val, x_val), axis=0)
    # y_train_val = np.concatenate((y_train_val, y_val), axis=0)
    # print(x_train_val.shape, type(x_train_val))
    # print(y_train_val.shape, type(y_train_val))
    # return

    fixed_model = model

    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')  
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed_{}_{}.h5'.format(adjustment_strategy, activation))
    log_path = os.path.join(model_weights_dir, 'log_{}.txt'.format(adjustment_strategy))

    if not os.path.exists(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)

    datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    logger('----------original model----------', log_path)

    # submodels 
    _, base_train_acc = fixed_model.evaluate(x_train_val, y_train_val)
    logger('The train accuracy: {:.4f}'.format(base_train_acc), log_path)
    _, base_val_acc = fixed_model.evaluate(x_val, y_val)
    # print('The validation accuracy: {:.4f}'.format(base_val_acc))
    logger('The validation accuracy: {:.4f}'.format(base_val_acc), log_path)
    _, base_test_acc = fixed_model.evaluate(x_test, y_test)
    # print('The test accuracy: {:.4f}'.format(base_test_acc))
    logger('The test accuracy: {:.4f}'.format(base_test_acc), log_path)
    
    best_weights = fixed_model.get_weights()
    best_acc = base_val_acc

    # find all indices of xs that original model fails on them.
    # fail_xs, fail_ys, fail_ys_label, fail_num = get_failing_cases(fixed_model, x_train_val, y_train_val)
    fail_xs, fail_ys, fail_ys_label, fail_num = get_failing_cases(fixed_model, x_train, y_train) # use the whole training dataset

    if settings.NUM_SUBMODELS == 20:
        sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_matrix_{}_{}.npy'.format(settings.RANDOM_SEED, settings.NUM_SUBMODELS))
    else:
        sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_matrix_{}_{}.npy'.format(settings.RANDOM_SEED, settings.NUM_SUBMODELS))
    sub_correct_matrix = None # 1: predicts correctly, -1: predicts incorrectly.
    print('obtaining sub correct matrix...')

    if not os.path.exists(sub_correct_matrix_path):
        # obtain submodel correctness matrix
        sub_correct_matrix = cal_sub_corr_matrix(fixed_model, sub_correct_matrix_path, submodel_dir, fail_xs, fail_ys, fail_ys_label, fail_num, num_submodels=20)
    else:
        sub_correct_matrix = np.load(sub_correct_matrix_path)

    sub_weights_list = get_submodels_weights(fixed_model, submodel_dir)
    print('collected.')
    fixed_model.load_weights(trained_weights_path)
    # print(sub_correct_matrix.shape)
    # print(sub_correct_matrix[0:20, :])

    # print('start fixing process...')
    logger('----------start fixing process----------', log_path)
    logger('number of cases to be adjusted: {}'.format(sub_correct_matrix.shape[0]), log_path)
    for _ in range(settings.LOOP_COUNT):
        np.random.shuffle(sub_correct_matrix)


        # load batches rather than single input.
        iter_num, rest = divmod(sub_correct_matrix.shape[0], settings.FIX_BATCH_SIZE)
        if rest != 0:
            iter_num += 1
        
        print('iter num: {}'.format(iter_num))
        # batch version
        for i in range(iter_num):
            curr_weights = fixed_model.get_weights()
            batch_corr_matrix = sub_correct_matrix[settings.FIX_BATCH_SIZE*i : settings.FIX_BATCH_SIZE*(i+1), :]
            # print('---------------------------------')
            # print(batch_corr_matrix)
            # print('---------------------------------')
            corr_w, incorr_w = batch_get_adjustment_weights(batch_corr_matrix, sub_weights_list, adjustment_strategy, curr_weights)
            # print(len(corr_w),len(incorr_w))
            print('calculating batch adjust weights...')
            # adjust_w = None
            # print(adjust_w)
            adjust_w = batch_adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
            # print(curr_weights[0][0])
            # print('----------')
            # print(adjust_w[0][0])
            fixed_model.set_weights(adjust_w)


        # counter = 0
        # for index in range(sub_correct_matrix.shape[0]):
        #     curr_weights = fixed_model.get_weights()
        #     corr_mat = sub_correct_matrix[index, :]
            
        #     print('obtaining correct and incorrect weights...')
        #     if adjustment_strategy <= 3:
        #         corr_w, incorr_w = get_adjustment_weights(corr_mat, sub_weights_list, adjustment_strategy)
        #         print('calculating adjust weights...')
        #         adjust_w = adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
        #     else: # lite version
        #         print('calculating adjust weights...')
        #         adjust_w = adjust_weights_func_lite(corr_mat, sub_weights_list, curr_weights)
            
        #     if adjust_w == -1:
        #         continue
        #     fixed_model.set_weights(adjust_w)
        #     counter +=1

        #     if counter != 20:
        #         continue
        #     else:
        #         counter = 0


            _, curr_acc = fixed_model.evaluate(x_val, y_val)
            print('After adjustment, the validation accuracy: {:.4f}'.format(curr_acc))

            if curr_acc > best_acc:
                best_acc = curr_acc
                fixed_model.save_weights(fixed_weights_path)

                if adjustment_strategy <=3:
                    # further training epochs.
                    checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
                    checkpoint.best = best_acc
                    hist = fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=settings.BATCH_SIZE), 
                                                steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                                validation_data=(x_val, y_val), 
                                                epochs=settings.FURTHER_ADJUSTMENT_EPOCHS, 
                                                callbacks=[checkpoint])

                    # for key in hist.history:
                    #     print(key)

                    fixed_model.load_weights(fixed_weights_path)

                    # eval the model
                    _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                    # _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
                    best_acc = val_acc

                    # print('validation accuracy after further training: {:.4f}'.format(test_acc))
                    logger('validation accuracy improved, after further training: {:.4f}'.format(val_acc), log_path)
                else:
                    logger('validation accuracy improved: {:.4f}'.format(best_acc), log_path)
            else:
                fixed_model.load_weights(fixed_weights_path)


    fixed_model.load_weights(fixed_weights_path)
    if adjustment_strategy > 3:
        # final training process.
        _, val_acc =  fixed_model.evaluate(x_val, y_val)
        checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
        checkpoint.best = val_acc

        fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=settings.BATCH_SIZE), 
                                                steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                                validation_data=(x_val, y_val), 
                                                epochs=20, 
                                                callbacks=[checkpoint])
        fixed_model.load_weights(fixed_weights_path)


    # final evaluation.
    _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
    logger('----------final evaluation----------', log_path)
    logger('test accuracy: {:.4f}'.format(test_acc), log_path)

    
def apricot_rnn():
    pass


def apricot2(model, model_weights_dir, dataset, adjustment_strategy, activation='binary'):
    """
    including Apricot and Apricot lite
    input:
        * dataset: [x_train_val, y_train_val, x_val, y_val, x_test, y_test]
    """
    # package the dataset
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    # x_train_val = np.concatenate((x_train_val, x_val), axis=0)
    # y_train_val = np.concatenate((y_train_val, y_val), axis=0)
    # print(x_train_val.shape, type(x_train_val))
    # print(y_train_val.shape, type(y_train_val))
    # return

    fixed_model = model

    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')  
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed_{}_{}.h5'.format(adjustment_strategy, activation))
    log_path = os.path.join(model_weights_dir, 'log_{}.txt'.format(adjustment_strategy))

    if not os.path.exists(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)

    datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    logger('----------original model----------', log_path)

    # submodels 
    _, base_train_acc = fixed_model.evaluate(x_train_val, y_train_val)
    logger('The train accuracy: {:.4f}'.format(base_train_acc), log_path)
    _, base_val_acc = fixed_model.evaluate(x_val, y_val)
    # print('The validation accuracy: {:.4f}'.format(base_val_acc))
    logger('The validation accuracy: {:.4f}'.format(base_val_acc), log_path)
    _, base_test_acc = fixed_model.evaluate(x_test, y_test)
    # print('The test accuracy: {:.4f}'.format(base_test_acc))
    logger('The test accuracy: {:.4f}'.format(base_test_acc), log_path)
    
    best_weights = fixed_model.get_weights()
    best_acc = base_val_acc

    # find all indices of xs that original model fails on them.
    # fail_xs, fail_ys, fail_ys_label, fail_num = get_failing_cases(fixed_model, x_train_val, y_train_val)
    fail_xs, fail_ys, fail_ys_label, fail_num = get_failing_cases(fixed_model, x_train, y_train) # use the whole training dataset

    if settings.NUM_SUBMODELS == 20:
        sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_matrix_{}_{}.npy'.format(settings.RANDOM_SEED, settings.NUM_SUBMODELS))
    else:
        sub_correct_matrix_path = os.path.join(model_weights_dir, 'corr_matrix_{}_{}.npy'.format(settings.RANDOM_SEED, settings.NUM_SUBMODELS))
    sub_correct_matrix = None # 1: predicts correctly, -1: predicts incorrectly.
    print('obtaining sub correct matrix...')

    if not os.path.exists(sub_correct_matrix_path):
        # obtain submodel correctness matrix
        sub_correct_matrix = cal_sub_corr_matrix(fixed_model, sub_correct_matrix_path, submodel_dir, fail_xs, fail_ys, fail_ys_label, fail_num, num_submodels=20)
    else:
        sub_correct_matrix = np.load(sub_correct_matrix_path)

    sub_weights_list = get_submodels_weights(fixed_model, submodel_dir)
    print('collected.')
    fixed_model.load_weights(trained_weights_path)
    # print(sub_correct_matrix.shape)
    # print(sub_correct_matrix[0:20, :])

    # print('start fixing process...')
    logger('----------start fixing process----------', log_path)
    logger('number of cases to be adjusted: {}'.format(sub_correct_matrix.shape[0]), log_path)
    for _ in range(settings.LOOP_COUNT):
        np.random.shuffle(sub_correct_matrix)


        # # load batches rather than single input.
        # iter_num, rest = divmod(sub_correct_matrix.shape[0], settings.FIX_BATCH_SIZE)
        # if rest != 0:
        #     iter_num += 1
        
        # print('iter num: {}'.format(iter_num))
        # # batch version
        # for i in range(iter_num):
        #     curr_weights = fixed_model.get_weights()
        #     batch_corr_matrix = sub_correct_matrix[settings.FIX_BATCH_SIZE*i : settings.FIX_BATCH_SIZE*(i+1), :]
        #     # print('---------------------------------')
        #     # print(batch_corr_matrix)
        #     # print('---------------------------------')
        #     corr_w, incorr_w = batch_get_adjustment_weights(batch_corr_matrix, sub_weights_list, adjustment_strategy, curr_weights)
        #     # print(len(corr_w),len(incorr_w))
        #     print('calculating batch adjust weights...')
        #     # adjust_w = None
        #     # print(adjust_w)
        #     adjust_w = batch_adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
        #     # print(curr_weights[0][0])
        #     # print('----------')
        #     # print(adjust_w[0][0])
        #     fixed_model.set_weights(adjust_w)


        counter = 0
        for index in range(sub_correct_matrix.shape[0]):
            curr_weights = fixed_model.get_weights()
            corr_mat = sub_correct_matrix[index, :]
            
            print('obtaining correct and incorrect weights...')
            if adjustment_strategy <= 3:
                corr_w, incorr_w = get_adjustment_weights(corr_mat, sub_weights_list, adjustment_strategy)
                print('calculating adjust weights...')
                adjust_w = adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
            else: # lite version
                print('calculating adjust weights...')
                adjust_w = adjust_weights_func_lite(corr_mat, sub_weights_list, curr_weights)
            
            if adjust_w == -1:
                continue
            fixed_model.set_weights(adjust_w)
            counter +=1

            if counter != 20:
                continue
            else:
                counter = 0


            _, curr_acc = fixed_model.evaluate(x_val, y_val)
            print('After adjustment, the validation accuracy: {:.4f}'.format(curr_acc))

            if curr_acc > best_acc:
                best_acc = curr_acc
                fixed_model.save_weights(fixed_weights_path)

                if adjustment_strategy <=3:
                    # further training epochs.
                    checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
                    checkpoint.best = best_acc
                    hist = fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=settings.BATCH_SIZE), 
                                                steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                                validation_data=(x_val, y_val), 
                                                epochs=settings.FURTHER_ADJUSTMENT_EPOCHS, 
                                                callbacks=[checkpoint])

                    # for key in hist.history:
                    #     print(key)

                    fixed_model.load_weights(fixed_weights_path)

                    # eval the model
                    _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                    # _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
                    best_acc = val_acc

                    # print('validation accuracy after further training: {:.4f}'.format(test_acc))
                    logger('validation accuracy improved, after further training: {:.4f}'.format(val_acc), log_path)
                else:
                    logger('validation accuracy improved: {:.4f}'.format(best_acc), log_path)
            else:
                fixed_model.load_weights(fixed_weights_path)


    fixed_model.load_weights(fixed_weights_path)
    if adjustment_strategy > 3:
        # final training process.
        _, val_acc =  fixed_model.evaluate(x_val, y_val)
        checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
        checkpoint.best = val_acc

        fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=settings.BATCH_SIZE), 
                                                steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                                validation_data=(x_val, y_val), 
                                                epochs=20, 
                                                callbacks=[checkpoint])
        fixed_model.load_weights(fixed_weights_path)


    # final evaluation.
    _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
    logger('----------final evaluation----------', log_path)
    logger('test accuracy: {:.4f}'.format(test_acc), log_path)

