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
from ApricotFamily.ApricotPlus.func import batch_get_adjust_w
from datetime import datetime
import copy

def apricorn(model, model_weights_dir, dataset):
    """
    the basic idea of apricorn is to update rDLMs at the same time.
    """
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    train_size = len(x_train)
    val_size = len(x_train_val)
    test_size = len(x_test)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'apricorn_fixed.h5')
    log_path = os.path.join(model_weights_dir, 'apricorn.log')

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    fixed_model.load_weights(trained_weights_path)
    start = datetime.now()

    sep_num = 5
    sep_count = 0

    logger('---------------original model---------------', log_path)
    # region initialization
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
        sub_correct_mat = apricot_cal_sub_corr_mat(fixed_model, submodel_dir, fail_xs, fail_ys,
                                                   num_submodels=NUM_SUBMODELS)
        np.save(sub_correct_matrix_path, sub_correct_mat)
    else:
        print('loading matrix...')
        sub_correct_mat = np.load(sub_correct_matrix_path)

    fixed_model.load_weights(trained_weights_path)

    weights_list = get_weights_list(fixed_model, submodel_dir, NUM_SUBMODELS)

    best_train_acc = base_train_acc
    best_val_acc = base_val_acc
    best_test_acc = base_test_acc

    # if not os.path.exists(fixed_weights_path):
    fixed_model.save_weights(fixed_weights_path)
    # endregion

    # reduce the sub_correct_mat
    sub_correct_mat, sorted_idx, select_num = reduce_sub_corr_mat(sub_correct_mat, rate=0.1)

    origin_sub_correct_mat = copy.deepcopy(sub_correct_mat)

    # Apricorn: iterates all failing cases.
    FIX_BATCH_SIZE = 1
    update_all = False
    impr_count = 0
    start = datetime.now()
    for count in range(1):
        # for i in range()
        iter_count, res = divmod(sub_correct_mat.shape[0], FIX_BATCH_SIZE)
        if res != 0:
            iter_count += 1
        print(iter_count)
        for i in range(iter_count):
            fixed_model.load_weights(fixed_weights_path)
            curr_w = fixed_model.get_weights()
            batch_corr_mat = sub_correct_mat[FIX_BATCH_SIZE * i: FIX_BATCH_SIZE * (i + 1)]
            adjust_w, adj_index_list = apricorn_batch_adjust_w(curr_w, batch_corr_mat, weights_list)  # update in lite way.
            # adjust_w = batch_get_adjust_w(curr_w, batch_corr_mat, weights_list)  # update in plus way.

            fixed_model.set_weights(adjust_w)

            x = int(count * sub_correct_mat.shape[0] + i + 1)
            y = int(sub_correct_mat.shape[0])
            _, curr_acc = fixed_model.evaluate(x_val, y_val)
            print('[iteration {}/{}] current val acc: {:.4f}, best val acc: {:.4f}'.format(x, y, curr_acc, best_val_acc))

            if curr_acc > best_val_acc:
                best_val_acc = curr_acc
                fixed_model.save_weights(fixed_weights_path)
                logger('Improved. val acc: {:.4f}'.format(best_val_acc), log_path)

                sep_count += 1
                if sep_count <= sep_num:  # reduce the number of training.
                    # sep_count = 0
                    # train the fixed model.
                    checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                 mode='max')
                    checkpoint.best = best_val_acc
                    hist = fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE),
                                                     steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1,
                                                     validation_data=(x_val, y_val),
                                                     epochs=3,  # 3 epochs
                                                     callbacks=[checkpoint])
                    fixed_model.load_weights(fixed_weights_path)
                    temp_val_acc = np.max(np.array(hist.history['val_accuracy']))
                else:
                    temp_val_acc = best_val_acc

                curr_w = fixed_model.get_weights()

                if temp_val_acc > best_val_acc:
                    # val acc improved.
                    best_val_acc = temp_val_acc
                # Apricorn: update weights list.
                # print('update weights list...')
                # # prepare the train
                # weights_list, sub_correct_mat = apricorn_update_weights_list(fixed_model, curr_w, batch_corr_mat, weights_list,
                #                                                              adj_index_list=adj_index_list,
                #                                                              datagen=datagen,
                #                                                              x_val=x_val,
                #                                                              y_val=y_val,
                #                                                              x_train_val=x_train_val,
                #                                                              y_train_val=y_train_val,
                #                                                              sub_correct_mat=sub_correct_mat,
                #                                                              fail_xs=fail_xs,
                #                                                              fail_ys=fail_ys,
                #                                                              index=sorted_idx,
                #                                                              num=select_num,
                #                                                              update_all=update_all)  # lr=0.01

            else:
                impr_count += 1
                if impr_count == 10:
                    impr_count = 0
                    update_all = True
                fixed_model.load_weights(fixed_weights_path)

            print('update weights list...')
            # prepare the train
            weights_list, sub_correct_mat = apricorn_update_weights_list(fixed_model, curr_w, batch_corr_mat,
                                                                         weights_list,
                                                                         adj_index_list=adj_index_list,
                                                                         datagen=datagen,
                                                                         x_val=x_val,
                                                                         y_val=y_val,
                                                                         x_train_val=x_train_val,
                                                                         y_train_val=y_train_val,
                                                                         sub_correct_mat=sub_correct_mat,
                                                                         fail_xs=fail_xs,
                                                                         fail_ys=fail_ys,
                                                                         index=sorted_idx,
                                                                         num=select_num)  # lr=0.01

        # sub_correct_mat = copy.deepcopy(origin_sub_correct_mat)
        # np.random.shuffle(sub_correct_mat)

    end = datetime.now()
    logger('Spend time: {}'.format(end - start), log_path)
