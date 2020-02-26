"""
Algorithms of Apricot+ and Apricot+ Lite

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

from .Apricot_utils import *
from settings import *
from general import *
import os
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input
from datetime import datetime

from model import *


def train_model(model, model_name, num_submodels, subset_size, dataset='cifar10', ver=1, train_sub=False, random_seed=42):
    model_dir_path = None
    if 'cifar' in dataset or 'imagenet' in dataset:
        model_dir_path = os.path.join(WEIGHTS_DIR, 'CNN')
        model_dir_path = os.path.join(model_dir_path, model_name)
        model_dir_path = os.path.join(model_dir_path, '{}'.format(ver))

        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
    else:
        # TODO: RNN model or else
        pass

    if not os.path.exists(os.path.join(model_dir_path, 'initialized.h5')):
        model.save_weights(os.path.join(model_dir_path, 'initialized.h5'))
    else:
        model.load_weights(os.path.join(model_dir_path, 'initialized.h5'))

    # load dataset
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = train_test_split(x_train, y_train, test_size=VAL_RATE, random_state=random_seed)

    f = open(os.path.join(model_dir_path, 'seed.txt'),'w')
    f.write(str(random_seed))
    f.close()

    # pre-training step
    if not os.path.exists(os.path.join(model_dir_path, 'pretrained.h5')):
        datagen = ImageDataGenerator(horizontal_flip=True,
                                width_shift_range=0.125,
                                height_shift_range=0.125,
                                fill_mode='constant', cval=0.)
        datagen.fit(x_train_val)
        
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                                        steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                        validation_data=(x_val, y_val), 
                                        epochs=PRE_EPOCHS)
        model.save_weights(os.path.join(model_dir_path, 'pretrained.h5'))
    else:
        model.load_weights(os.path.join(model_dir_path, 'pretrained.h5'))


    if not os.path.exists(os.path.join(model_dir_path, 'submodels')):
        os.makedirs(os.path.join(model_dir_path, 'submodels'))
    datagen = ImageDataGenerator(horizontal_flip=True,
                                width_shift_range=0.125,
                                height_shift_range=0.125,
                                fill_mode='constant', cval=0.)
    datagen.fit(x_train_val)


    # submodel training step
    if train_sub:  # whether train the submodels
        step = int((x_train.shape[0] - subset_size) / num_submodels)
        for i in range(num_submodels):
            # subset
            sub_weights_path = os.path.join(os.path.join(model_dir_path, 'submodels'))
            sub_weights_path = os.path.join(sub_weights_path, 'sub_{}.h5'.format(i))

            sub_x_train_val = x_train_val[step*i : subset_size + step*i]
            sub_y_train_val = y_train_val[step*i : subset_size + step*i]

            model.load_weights(os.path.join(model_dir_path, 'pretrained.h5'))
            model.fit_generator(datagen.flow(sub_x_train_val, sub_y_train_val, batch_size=BATCH_SIZE), 
                                            steps_per_epoch=len(sub_x_train_val) // BATCH_SIZE + 1, 
                                            validation_data=(x_val, y_val), 
                                            epochs=SUB_EPOCHS)
            model.save_weights(sub_weights_path)

    # original DL model training step
    if not os.path.exists(os.path.join(model_dir_path, 'trained.h5')):
        model.load_weights(os.path.join(model_dir_path, 'pretrained.h5'))
        model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                            validation_data=(x_val, y_val), 
                                            epochs=AFTER_EPOCHS)
        model.save_weights(os.path.join(model_dir_path, 'trained.h5'))
  

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
        else:
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


def adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation='binary'):
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
    adjust_weights = -1
    if adjustment_strategy == 1:
        if corr_w is None or incorr_w is None:
            return -1
        else:
            adjust_weights = [item[0] - learning_rate * (item[0] - item[1]) + learning_rate * (item[0] - item[2]) for item in zip(curr_weights, corr_w, incorr_w)]
    if adjustment_strategy == 2:
        if corr_w is None:
            return -1
        else:
            adjust_weights = [item[0] - learning_rate * (item[0] - item[1]) for item in zip(curr_weights, corr_w)]
    if adjustment_strategy == 3:
        if incorr_w is None:
            return -1
        else:
            adjust_weights = [item[0] + learning_rate * (item[0] - item[1]) for item in zip(curr_weights, incorr_w)]
            
    if adjustment_strategy == 4:
        if corr_w is None or incorr_w is None:
            return -1
        else:
            diff_corr_w = get_difference_func(curr_weights, corr_w)
            diff_incorr_w = get_difference_func(curr_weights, incorr_w)
            adjust_weights = [item[0] - learning_rate * np.multiply(item[0], item[1]) + learning_rate * np.multiply(item[0], item[2]) for item in zip(curr_weights, diff_corr_w, diff_incorr_w)]            
    
    if adjustment_strategy == 5:
        if corr_w is None:
            return -1
        else:
            diff_corr_w = get_difference_func(curr_weights, corr_w)
            adjust_weights = [item[0] - learning_rate * np.multiply(item[0], item[1]) for item in zip(curr_weights, diff_corr_w)]
    
    if adjustment_strategy == 6:
        if incorr_w is None:
            return -1
        else:
            diff_incorr_w = get_difference_func(curr_weights, incorr_w)
            adjust_weights = [item[0] + learning_rate * np.multiply(item[0], item[1]) for item in zip(curr_weights, diff_incorr_w)]
    
    return adjust_weights


def apricot_plus_lite(model, model_name, get_trained_weights, x_train_val, y_train_val, x_val, y_val, x_test, y_test, adjustment_strategy, activation='binary', ver=1, dataset='cifar10', max_count=1, loop_count=100000, random_seed=42):
    weights_dir = os.path.join(WEIGHTS_DIR, 'CNN')
    weights_dir = os.path.join(weights_dir, model_name)
    weights_dir = os.path.join(weights_dir, '{}'.format(ver))
    
    # create the dir
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    if get_trained_weights:
        model.load_weights(os.path.join(weights_dir, 'trained.h5'))
    
    weights_after_dir = os.path.join(weights_dir, 'fixed_{}_{}.h5'.format(adjustment_strategy, activation))
    
    if not os.path.exists(weights_after_dir):
        model.save_weights(weights_after_dir)
    
    datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)

    datagen.fit(x_train_val)
    
    # build the fixed model.
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

    fixed_model.load_weights(os.path.join(weights_dir, 'trained.h5'))
    # fixed_model = copy.deepcopy(model)
    
    # evaluate the acc before fixing.
    print('----------origin model----------')
    if dataset in ['cifar100', 'imagenet']:
        _, acc_top_1_train, train_acc = fixed_model.evaluate(x_train_val, y_train_val)
        print('[==log==] training acc. before fixing: top-1: {:.4f}, top-5: {:.4f}'.format(acc_top_1_train, train_acc))
        _, acc_top_1_val, origin_acc = fixed_model.evaluate(x_val, y_val)
        print('[==log==] validation acc. before fixing: top-1: {:.4f}, top-5: {:.4f}'.format(acc_top_1_val, origin_acc))
        _, acc_top_1_test, test_acc = fixed_model.evaluate(x_test, y_test)
        print('[==log==] test acc. before fixing: top-1: {:.4f}, top-5: {:.4f}'.format(acc_top_1_test, test_acc))
        logger(weights_dir, '========================')
        logger(weights_dir, 'model: {}, adjustment strategy: {}, ver: {}'.format(model_name, adjustment_strategy, ver))
        logger(weights_dir, 'TOP-1: train acc.: {:4f}, val acc.: {:4f}, test acc.: {:4f}'.format(acc_top_1_train, acc_top_1_val, acc_top_1_test))
        logger(weights_dir, 'TOP-5: train acc.: {:4f}, val acc.: {:4f}, test acc.: {:4f}'.format(train_acc, origin_acc, test_acc))
        
    else:
        _, origin_acc = fixed_model.evaluate(x_val, y_val)
        print('----------origin model----------')
        _, train_acc = fixed_model.evaluate(x_train_val, y_train_val)
        print('[==log==] training acc. before fixing: {:.4f}'.format(train_acc))
        print('[==log==] validation acc. before fixing: {:.4f}'.format(origin_acc))
        _, test_acc = fixed_model.evaluate(x_test, y_test)
        print('[==log==] test acc. before fixing: {:.4f}'.format(test_acc))
        logger(weights_dir, '========================')
        logger(weights_dir, 'model: {}, adjustment strategy: {}, ver: {}'.format(model_name, adjustment_strategy, ver))
        logger(weights_dir, 'train acc.: {:4f}, val acc.: {:4f}, test acc.: {:4f}'.format(train_acc, origin_acc, test_acc))

    # start time
    start_time = datetime.now()

    # start fixing
    best_weights = fixed_model.get_weights()
    best_acc = origin_acc
    
    # find all indices of xs that original model fails on them.
    y_preds = model.predict(x_train_val)
    y_pred_label = np.argmax(y_preds, axis=1)
    y_true = np.argmax(y_train_val, axis=1)

    index_diff = np.nonzero(y_pred_label - y_true)

    fail_xs = x_train_val[index_diff]
    fail_ys = y_train_val[index_diff]
    fail_ys_label = np.argmax(fail_ys, axis=1)
    fail_num = int(np.size(index_diff))
 
    sub_correct_matrix_path = os.path.join(weights_dir, 'corr_matrix_{}.npy'.format(random_seed))
    sub_correct_matrix = None # 1: predicts correctly, 0: predicts incorrectly.
    
    sub_weights_list = None

    if not os.path.exists(sub_correct_matrix_path):
        # obtain submodel correctness matrix
        submodels_path = os.path.join(weights_dir, 'submodels')

        for root, dirs, files in os.walk(submodels_path):
            for f in files:
                temp_w_path = os.path.join(root, f)
                fixed_model.load_weights(temp_w_path)
                sub_y_pred = fixed_model.predict(fail_xs)

                # top-1 accuracy
                if not dataset in ['cifar100', 'imagenet']:
                    sub_col = np.argmax(sub_y_pred, axis=1) - fail_ys_label
                    sub_col[sub_col != 0] = 1
                # top-5 accuracy
                else:
                    sub_col = K.in_top_k(sub_y_pred, K.argmax(fail_ys, axis=-1), 5)
                    sub_col = K.get_value(sub_col)
                    sub_col = sub_col.astype(int)
                    sub_col = np.ones(shape=sub_col.shape) - sub_col

                if sub_correct_matrix is None:
                    sub_correct_matrix = sub_col.reshape(fail_num, 1)
                else:
                    sub_correct_matrix = np.concatenate((sub_correct_matrix, sub_col.reshape(fail_num, 1)), axis=1)

            sub_correct_matrix = np.ones(shape=sub_correct_matrix.shape) - sub_correct_matrix  # here change 0 to 1 (for correctly predicted case)
            np.save(sub_correct_matrix_path, sub_correct_matrix)

        # for sub in submodels:
        #     sub_y_pred = sub.predict(fail_xs)
        #     sub_col = np.argmax(sub_y_pred, axis=1) - fail_ys_label
        #     sub_col[sub_col != 0] = 1
        #     if sub_correct_matrix is None:
        #         sub_correct_matrix = copy.deepcopy(sub_col.reshape(fail_num, 1))
        #     else:
        #         sub_correct_matrix = np.concatenate((sub_correct_matrix, sub_col.reshape(fail_num, 1)), axis=1)
        # sub_correct_matrix = np.ones(shape=sub_correct_matrix.shape) - sub_correct_matrix
        # np.save(sub_correct_matrix_path, sub_correct_matrix)
    else:
        sub_correct_matrix = np.load(sub_correct_matrix_path)
        # revision
        sub_weights_list = get_submodels_weights(fixed_model, model_name, dataset, os.path.join(weights_dir, 'submodels'))
    
    # main loop
    fixed_model.load_weights(weights_after_dir)

    logger(weights_dir,'-----------------')
    logger(weights_dir,'adjustment strategy {}'.format(adjustment_strategy))
    logger(weights_dir,'LOOP_COUNT: {}, BATCH_SIZE: {}, learning_rate: {}'.format(loop_count, BATCH_SIZE, learning_rate))
    logger(weights_dir,'PRE_EPOCHS: {}, AFTER_EPOCHS: {}, SUB_EPOCHS: {}, MAX_COUNT: {}'.format(PRE_EPOCHS, AFTER_EPOCHS, SUB_EPOCHS, max_count))
    logger(weights_dir,'-----------------')

    for _ in range(loop_count):
        np.random.shuffle(sub_correct_matrix)
        iter_count = 0
        for index in range(sub_correct_matrix.shape[0]):

            if iter_count >= max_count:
                break

            curr_weights = fixed_model.get_weights()
            corr_mat = sub_correct_matrix[index, :]

            # lite version
            corr_w, incorr_w = get_adjustment_weights(corr_mat, sub_weights_list, adjustment_strategy)
            adjust_w = adjust_weights_func(curr_weights, corr_w, incorr_w, adjustment_strategy, activation=activation)
            
            if adjust_w == -1:
                continue
                
            fixed_model.set_weights(adjust_w)

            if not dataset in ['cifar100', 'imagenet']:
                _, curr_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
            else:
                _, _, curr_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
            print('tried times: {}, validation accuracy after adjustment: {:.4f}'.format(index, curr_acc))
            if curr_acc > best_acc:
                best_acc = curr_acc
                fixed_model.save_weights(weights_after_dir)

                if adjustment_strategy <=3:
                    # Apricot+ further training process
                    if not dataset in ['cifar100', 'imagenet']:
                        checkpoint = ModelCheckpoint(weights_after_dir, monitor=MONITOR, verbose=1, save_best_only=True, mode='max') 
                    else:
                        checkpoint = ModelCheckpoint(weights_after_dir, monitor='val_top_k_acc', verbose=1, save_best_only=True, mode='max') 

                    checkpoint.best = best_acc
                    fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                                            validation_data=(x_val, y_val), 
                                            epochs=FURTHER_ADJUSTMENT_EPOCHS, 
                                            callbacks=[checkpoint])
                    fixed_model.load_weights(weights_after_dir)
                    
                    if not dataset in ['cifar100', 'imagenet']:
                        _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                        _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
                    else:
                        _, _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                        _, _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)

                    print('validation acc. after retraining: {:.4f}'.format(val_acc))
                    print('test acc. after retraining: {:.4f}'.format(test_acc))
                    logger(weights_dir,'Improved, validation acc.: {:.4f}, test acc.:{:.4f}'.format(val_acc, test_acc))

                else:
                    print('-----------------------------')
                    print('evaluate on test dataset.')
                    best_acc = curr_acc
                    best_weights = adjust_w
                    fixed_model.save_weights(weights_after_dir)
                    # evaluation
                    if not dataset in ['cifar100', 'imagenet']:
                        _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                        _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
                    else:
                        _, _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
                        _, _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)
                    
                    print('validation acc. after retraining: {:.4f}'.format(val_acc))
                    print('test acc. after retraining: {:.4f}'.format(test_acc))
                    logger(weights_dir,'Improved, validation acc.: {:.4f}, test acc.:{:.4f}'.format(val_acc, test_acc))
                
            else:
                fixed_model.set_weights(best_weights)

            iter_count += 1
      
    # further training process.
    if not dataset in ['cifar100', 'imagenet']:
        checkpoint = ModelCheckpoint(weights_after_dir, monitor=MONITOR, verbose=1, save_best_only=True, mode='max') 
    else:
        checkpoint = ModelCheckpoint(weights_after_dir, monitor='val_top_k_acc', verbose=1, save_best_only=True, mode='max') 

    checkpoint.best = best_acc
    fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE), 
                            steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1, 
                            validation_data=(x_val, y_val), 
                            epochs=FURTHER_ADJUSTMENT_EPOCHS, 
                            callbacks=[checkpoint])
    
    # end time
    end_time = datetime.now()
    time_delta = end_time - start_time
    print('time used for adaptation: {}'.format(str(time_delta)))
    logger(weights_dir, 'time used for adaptation: {}'.format(str(time_delta)))

    fixed_model.load_weights(weights_after_dir)
    best_weights = fixed_model.get_weights()


    if dataset in ['cifar100', 'imagenet']:
        _, acc_top_1_train, train_acc = fixed_model.evaluate(x_train_val, y_train_val)
        _, acc_top_1_val, origin_acc = fixed_model.evaluate(x_val, y_val)
        _, acc_top_1_test, test_acc = fixed_model.evaluate(x_test, y_test)

        print('after adjustment and retraining, TOP-1 train acc.: {}, val acc.: {}, test acc.: {}'.format(acc_top_1_train, acc_top_1_val, acc_top_1_test))
        print('after adjustment and retraining, TOP-5 train acc.: {}, val acc.: {}, test acc.: {}'.format(train_acc, origin_acc, test_acc))
        
        logger(weights_dir, 'after adjustment and retraining, TOP-1 train acc.: {}, val acc.: {}, test acc.: {}'.format(acc_top_1_train, acc_top_1_val, acc_top_1_test))
        logger(weights_dir, 'after adjustment and retraining, TOP-5 train acc.: {}, val acc.: {}, test acc.: {}'.format(train_acc, origin_acc, test_acc))


    else:
        _, train_acc = fixed_model.evaluate(x_train_val, y_train_val, verbose=0)
        _, val_acc = fixed_model.evaluate(x_val, y_val, verbose=0)
        _, test_acc = fixed_model.evaluate(x_test, y_test, verbose=0)

        print('validation acc. after retraining: {:.4f}'.format(val_acc))
        print('test acc. after retraining: {:.4f}'.format(test_acc))

        logger(weights_dir, 'after adjustment and retraining, train acc.: {}, val acc.: {}, test acc.: {}'.format(train_acc, val_acc, test_acc))