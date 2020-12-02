import numpy as np
from ApricotFamily.Apricot.main import get_avg_weights, get_adjust_weights


def batch_get_adjust_w(curr_w, batch_corr_mat, weights_list, strategy):
    """
    0 represents failing case, 1 represents correct case.
    """
    adjust_w = curr_w
    for idx in range(batch_corr_mat.shape[0]):
        tmp_corr_mat = batch_corr_mat[idx]
        corr_avg, incorr_avg = get_avg_weights(tmp_corr_mat, weights_list)
        adjust_w = get_adjust_weights(adjust_w, corr_avg, incorr_avg, strategy)
    return adjust_w
