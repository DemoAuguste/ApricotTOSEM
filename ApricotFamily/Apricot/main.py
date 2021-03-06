# the main process of Apricot (conference version).
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils import load_dataset, split_validation_dataset
from utils import logger
from settings import NUM_SUBMODELS
from .func import *
from settings import BATCH_SIZE, FURTHER_ADJUSTMENT_EPOCHS
from keras import backend as K
from datetime import datetime


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
    log_path = os.path.join(model_weights_dir, 'apricot_{}.log'.format(adjustment_strategy))

    if not os.path.exists(fixed_weights_path):
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

    # base_train_acc, base_val_acc, base_test_acc = (1,1,1)  # test

    logger('train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(base_train_acc, base_val_acc, base_test_acc), log_path)

    # to simply the process, get the classification results of submodels first.
    # do not shuffle the training dataset.
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

    # iterates all training dataset.
    iter_batch_size = 20  # TODO revise hard-coding
    iter_num, ret = divmod(train_size, iter_batch_size)
    fail_idx_seq = get_formatted_batch_sequence(fail_index, total_num=train_size)  # binary indicator

    if ret != 0:
        iter_num += 1

    # the main process
    train_total_index = [i for i in range(train_size)]  # initialize indices for all training samples.
    train_total_index = np.array(train_total_index)

    sub_weights_list = get_weights_list(fixed_model, submodel_dir, num_submodels=NUM_SUBMODELS)

    fixed_model.load_weights(trained_weights_path)  # load the trained model.
    best_weights = fixed_model.get_weights()  # used for keeping the best weights of the model.
    best_train_acc = base_train_acc
    best_val_acc = base_val_acc
    best_test_acc = base_test_acc

    # if not os.path.exists(fixed_weights_path):
    fixed_model.save_weights(fixed_weights_path)

    # print('iteration: {}, number of failing cases: {}'.format(iter_num, len(fail_xs)))
    logger('iteration: {}, number of failing cases: {}'.format(iter_num, len(fail_xs)), log_path)

    print('start the main iteration process...')
    for i in range(iter_num):  # iterates by batch.
        try:
            # check if the index is in the fail_index.
            temp_train_index = train_total_index[i*iter_batch_size: (i+1)*iter_batch_size]
            temp_fail_idx_seq = fail_idx_seq[temp_train_index]  # temp binary indicator
            adjust_w = fixed_model.get_weights()
            # retrieve sub_correct_mat
            if np.sum(temp_fail_idx_seq) == 0:  # no failing cases.
                continue
            else:
                # exists failing cases.
                # get the failing case index
                # print(np.nonzero(temp_fail_idx_seq)[0])
                temp_fail_idx = temp_train_index[np.nonzero(temp_fail_idx_seq)[0]]
                print('[iteration {}/{}]'.format(i, iter_num), temp_fail_idx)  # Absolute index in train dataset
                for idx in temp_fail_idx:
                    sub_correct_mat_idx = int(np.sum(fail_idx_seq[:idx + 1]))  # mapping the total idx back to sub mat idx.
                    # print(sub_correct_mat_idx)
                    temp_sub_corr_mat = sub_correct_mat[sub_correct_mat_idx]

                    # adjust weights
                    corr_avg, incorr_avg = get_avg_weights(temp_sub_corr_mat, weights_list=sub_weights_list)
                    adjust_w = get_adjust_weights(adjust_w, corr_avg, incorr_avg, adjustment_strategy)

                # evaluation.
                fixed_model.set_weights(adjust_w)
                _, curr_acc = fixed_model.evaluate(x_val, y_val)
                print('After adjustment, the val acc: {:.4f}, best val acc: {:.4f}'.format(curr_acc, best_val_acc))

                if curr_acc > best_val_acc:
                    best_val_acc = curr_acc
                    fixed_model.save_weights(fixed_weights_path)
                    # further training process.
                    checkpoint = ModelCheckpoint(fixed_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
                    checkpoint.best = best_val_acc
                    hist = fixed_model.fit_generator(datagen.flow(x_train_val, y_train_val, batch_size=BATCH_SIZE),
                                                     steps_per_epoch=len(x_train_val) // BATCH_SIZE + 1,
                                                     validation_data=(x_val, y_val),
                                                     epochs=20,  # 3 epochs
                                                     callbacks=[checkpoint])
                    fixed_model.load_weights(fixed_weights_path)

                    temp_val_acc = np.max(np.array(hist.history['val_accuracy']))
                    temp_train_acc = np.max(np.array(hist.history['accuracy']))
                    if temp_val_acc > best_val_acc:
                        # val acc improved.
                        best_val_acc = temp_val_acc
                        best_train_acc = temp_train_acc
                    _, best_test_acc = fixed_model.evaluate(x_test, y_test)
                    # print(best_train_acc, best_val_acc)
                    logger('Improved. Train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}'.format(best_train_acc, best_val_acc, best_test_acc), log_path)
                    # fixed_model.save_weights(fixed_weights_path)
                else:  # worse than the best, rollback to the best case.
                    fixed_model.load_weights(fixed_weights_path)
        except:
            continue

    end = datetime.now()
    logger('Spend time: {}'.format(end - start), log_path)











