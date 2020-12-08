import numpy as np


def apricorn_batch_adjust_w(curr_w, batch_corr_mat, weights_list, lr=0.01):
    """
    same as apricot lite.
    """
    adjust_w = curr_w
    # region apricot lite adjustment process
    for i in range(batch_corr_mat.shape[0]):
        temp_corr_mat = batch_corr_mat[i]
        temp_incorr_mat = np.ones(temp_corr_mat.shape) - temp_corr_mat
        # randomly choose one.
        corr_w = adjust_w
        incorr_w = adjust_w
        if len(np.nonzero(temp_corr_mat)[0]) != 0:
            corr_idx = np.nonzero(temp_corr_mat)[0]
            # print(corr_idx)
            corr_idx = corr_idx[np.random.randint(corr_idx.shape[0], size=1)]
            # print(corr_idx)
            corr_w = weights_list[corr_idx[0]]
        if len(np.nonzero(temp_incorr_mat)[0]) != 0:
            incorr_idx = np.nonzero(temp_incorr_mat)[0]
            # print(incorr_idx)
            # print(incorr_idx.shape)
            incorr_idx = incorr_idx[np.random.randint(incorr_idx.shape[0], size=1)]
            incorr_w = weights_list[incorr_idx[0]]

        diff_corr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, corr_w)]
        diff_incorr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, incorr_w)]

        adjust_w = [item[0] - item[1] + item[2] for item in zip(curr_w, diff_corr_w, diff_incorr_w)]
    # endregion

    return adjust_w


def apricorn_update_weights_list(curr_w, batch_corr_mat, weights_list, lr=0.01):
    for i in range(batch_corr_mat.shape[0]):
        temp_corr_mat = batch_corr_mat[i]
        temp_incorr_mat = np.ones(temp_corr_mat.shape) - temp_corr_mat

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

    return weights_list
