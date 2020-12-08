import numpy as np
from settings import BATCH_SIZE

def apricorn_batch_adjust_w(curr_w, batch_corr_mat, weights_list, lr=0.01):
    """
    same as apricot lite.
    """
    adjust_w = curr_w
    adj_index_list = []
    # region apricot lite adjustment process
    for i in range(batch_corr_mat.shape[0]):
        temp_corr_mat = batch_corr_mat[i]
        temp_incorr_mat = np.ones(temp_corr_mat.shape) - temp_corr_mat
        # randomly choose one.
        corr_w = adjust_w
        incorr_w = adjust_w
        tmp_idx = [-1, -1]
        if len(np.nonzero(temp_corr_mat)[0]) != 0:
            corr_idx = np.nonzero(temp_corr_mat)[0]
            # print(corr_idx)
            corr_idx = corr_idx[np.random.randint(corr_idx.shape[0], size=1)]
            tmp_idx[0] = corr_idx
            # print(corr_idx)
            corr_w = weights_list[corr_idx[0]]
        if len(np.nonzero(temp_incorr_mat)[0]) != 0:
            incorr_idx = np.nonzero(temp_incorr_mat)[0]
            # print(incorr_idx)
            # print(incorr_idx.shape)
            incorr_idx = incorr_idx[np.random.randint(incorr_idx.shape[0], size=1)]
            tmp_idx[1] = incorr_idx
            incorr_w = weights_list[incorr_idx[0]]
        adj_index_list.append(tmp_idx)

        diff_corr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, corr_w)]
        diff_incorr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, incorr_w)]

        adjust_w = [item[0] - item[1] + item[2] for item in zip(curr_w, diff_corr_w, diff_incorr_w)]
    # endregion

    return adjust_w, adj_index_list


def apricorn_update_weights_list(model, curr_w, batch_corr_mat, weights_list, adj_index_list=None, lr=0.01, **kwargs):
    sub_mat = kwargs['sub_correct_mat']
    for i in range(batch_corr_mat.shape[0]):
        temp_corr_mat = batch_corr_mat[i]
        temp_incorr_mat = np.ones(temp_corr_mat.shape) - temp_corr_mat

        if adj_index_list is None:
            # try: updates incorrect weights.
            if len(np.nonzero(temp_incorr_mat)[0]) != 0:
                # updates weights list
                incorr_idx = np.nonzero(temp_incorr_mat)[0]
                print('index of updated weights: ', end='')
                print(incorr_idx)
                for idx in incorr_idx:
                    temp_diff_w = [lr * (item[0] - item[1]) for item in zip(weights_list[idx], curr_w)]
                    weights_list[idx] = [item[0] - item[1] for item in zip( weights_list[idx], temp_diff_w)]
            else:
                continue
        else:
            for bat_idx in adj_index_list:
                incorr_idx = bat_idx[1]
                print(incorr_idx)
                if incorr_idx == -1:
                    continue

                # prepare the data.
                temp_idx = np.random.randint(kwargs['x_train_val'].shape[0], size=10000)
                temp_x_train = kwargs['x_train_val'][temp_idx]
                temp_y_train = kwargs['y_train_val'][temp_idx]
                
                # update
                model.set_weights(weights_list[int(incorr_idx)])
                model.fit_generator(kwargs['datagen'].flow(temp_x_train, temp_y_train, batch_size=BATCH_SIZE),
                                    steps_per_epoch=len(temp_x_train) // BATCH_SIZE + 1,
                                    validation_data=(kwargs['x_val'], kwargs['y_val']),
                                    epochs=1)  # 3 epochs)

                # update sub_correct_mat
                pred_ys = model.predict(kwargs['fail_xs'])
                pred_ys_label = np.argmax(pred_ys, axis=1)
                fail_ys_label = np.argmax(kwargs['fail_ys'], axis=1)
                temp_col = pred_ys_label == fail_ys_label
                temp_col = np.array(temp_col, dtype=np.int)
                print(temp_col.shape)
                print(sub_mat.shape)
                print(incorr_idx)
                sub_mat[int(incorr_idx)] = temp_col

    return weights_list, sub_mat
