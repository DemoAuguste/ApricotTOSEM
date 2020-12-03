import numpy as np


def batch_lite_get_adjust_w(curr_w, batch_corr_mat, weights_list, lr=0.0005):
    """
    0 represents failing case, 1 represents correct case.
    """
    # get two sets
    adjust_w = curr_w
    for i in range(batch_corr_mat.shape[0]):
        temp_corr_mat = batch_corr_mat[i]
        temp_incorr_mat = np.ones(temp_corr_mat.shape) - temp_corr_mat
        # randomly choose one.
        corr_w = adjust_w
        incorr_w = adjust_w
        if len(temp_corr_mat) != 0:
            corr_idx = np.nonzero(temp_corr_mat)[0]
            corr_idx = corr_idx[np.random.randint(corr_idx.shape[0], size=1)]
            print(corr_idx)
            corr_w = weights_list[corr_idx]
        if len(temp_incorr_mat) != 0:
            incorr_idx = np.nonzero(temp_incorr_mat)[0]
            incorr_idx = incorr_idx[np.random.randint(incorr_idx.shape[0], size=1)]
            incorr_w = weights_list[incorr_idx]

        diff_corr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, corr_w)]
        diff_incorr_w = [lr * (item[0] - item[1]) for item in zip(adjust_w, incorr_w)]

        adjust_w = [item[0] - item[1] + item[2] for item in zip(curr_w, diff_corr_w, diff_incorr_w)]

    return adjust_w
