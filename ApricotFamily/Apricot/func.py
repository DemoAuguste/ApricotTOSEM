import os
import numpy as np
import copy
from settings import learning_rate


def get_formatted_batch_sequence(fail_index, total_num):
    """
    total_num: the total number of the training dataset.
    fail_index: identified failing cases.
    """
    fail_idx_seq = np.zeros(total_num)
    fail_idx_seq[fail_index] = 1

    return fail_idx_seq


def get_indexed_failing_cases(model, xs, ys):
    """
    Given a DL model, get the failing cases with indices.
    """
    y_preds = model.predict(xs)
    y_pred_label = np.argmax(y_preds, axis=1)
    y_true = np.argmax(ys, axis=1)

    fail_index = np.nonzero(y_pred_label - y_true)[0]
    fail_xs = xs[fail_index]
    fail_ys = ys[fail_index]
    fail_ys_label = np.argmax(fail_ys, axis=1)
    fail_num = int(np.size(fail_index))

    return fail_xs, fail_ys, fail_ys_label, fail_num, fail_index


def get_class_prob_mat(model, xs, ys):
    pred_prob_mat = model.predict(xs)

    max_ind_mat = list(map(lambda x: x == max(x), pred_prob_mat)) * np.ones(shape=pred_prob_mat.shape)
    max_prob_mat = pred_prob_mat * max_ind_mat * ys
    sum_prob_mat = np.sum(max_prob_mat, axis=0)
    class_sum_mat = np.sum(max_ind_mat, axis=0)
    class_prob_mat = sum_prob_mat / class_sum_mat

    # print(class_prob_mat)
    return class_prob_mat


def _compare(x,y):
    if max(x) > max(y):
        return x
    else:
        return np.zeros(shape=x.shape)


def apricot_cal_sub_corr_mat(model, submodels_path, fail_xs, fail_ys, num_submodels=20):
    """
    submodel binary indicator.
    0 represents failing case, 1 represents correct case.
    """
    sub_corr_mat = None
    for root, dirs, files in os.walk(submodels_path):
        for i in range(num_submodels):
            temp_w_path = os.path.join(root, 'sub_{}.h5'.format(i))
            model.load_weights(temp_w_path)
            pred_ys = model.predict(fail_xs)
            pred_ys_label = np.argmax(pred_ys, axis=1)
            fail_ys_label = np.argmax(fail_ys, axis=1)
            temp_col = pred_ys_label == fail_ys_label
            temp_col = np.array(temp_col, dtype=np.int)
            temp_col = temp_col[:, np.newaxis]
            if sub_corr_mat is None:
                sub_corr_mat = temp_col
            else:
                sub_corr_mat = np.concatenate((sub_corr_mat, temp_col), axis=1)
    # reverse the 1 and 0
    ret = np.ones(sub_corr_mat.shape)
    ret = ret - sub_corr_mat
    return ret


def get_weights_list(model, sub_path, num_submodels=20):
    ret = []
    temp_model = copy.deepcopy(model)
    for i in range(num_submodels):
        temp_weights_path = os.path.join(sub_path, 'sub_{}.h5'.format(i))
        temp_model.load_weights(temp_weights_path)
        ret.append(temp_model.get_weights())
    return ret


def cal_avg(weights_list):
    """
    ----------------------
    calculate the average weights of a weights list.
    ----------------------
    return:
    average weights of weights list. format: equals to model.get_weights()
    """
    sum_w = None
    total_num = len(weights_list)

    def weights_add(sum_w, w):
        if sum_w is None:
            sum_w = copy.deepcopy(w)
        else:
            sum_w = [sum(i) for i in zip(sum_w, w)]
        return sum_w

    for w in weights_list:
        sum_w = weights_add(sum_w, w)
    sum_w = [item / total_num for item in sum_w]

    return sum_w


def get_avg_weights(sub_corr_mat, weights_list):
    corr_idx = np.nonzero(sub_corr_mat)[0]  # incorrect index
    corr_list = []
    incorr_list = []
    for i in range(len(weights_list)):
        if i in corr_idx.tolist():
            corr_list.append(weights_list[i])
        else:
            incorr_list.append(weights_list[i])
    corr_avg = []
    incorr_avg = []
    if len(corr_list) != 0:
        corr_avg = cal_avg(corr_list)
    if len(incorr_list) != 0:
        incorr_avg = cal_avg(incorr_list)
    return corr_avg, incorr_avg


def get_adjust_weights(curr_w, corr_avg, incorr_avg, strategy):
    """
    adjust weights by a given strategy
    """
    adjust_w = None
    if len(corr_avg) == 0:
        corr_w = curr_w
    else:
        corr_w = corr_avg
    if len(incorr_avg) == 0:
        incorr_w = curr_w
    else:
        incorr_w = incorr_avg

    diff_corr_w = [learning_rate * (item[0] - item[1]) for item in zip(curr_w, corr_w)]
    diff_incorr_w = [learning_rate * (item[0] - item[1]) for item in zip(curr_w, incorr_w)]
    if strategy == 1:
        adjust_w = [item[0] - item[1] + item[2] for item in zip(curr_w, diff_corr_w, diff_incorr_w)]
    elif strategy == 2:
        adjust_w = [item[0] - item[1] for item in zip(curr_w, diff_corr_w)]
    elif strategy == 3:
        adjust_w = [item[0] + item[1] for item in zip(curr_w, diff_incorr_w)]
    else:
        NotImplementedError('Not implemented.')
    return adjust_w


def get_model_correct_mat(model, xs, ys, class_prob_mat):
    pred_prob_mat = model.predict(xs)
    # class_prob_mat = get_class_prob_mat(model, xs, ys) # threshold

    max_ind_mat = list(map(lambda x: x == max(x), pred_prob_mat)) * np.ones(shape=pred_prob_mat.shape)
    correct_ind_mat = max_ind_mat * ys  # if model predicts correctly, then one row should have one element equals to 1
    correct_pred_prob_mat = pred_prob_mat * correct_ind_mat

    # construct a threshold matrix
    threshold_mat = np.tile(class_prob_mat, (len(xs), 1))
    threshold_mat = threshold_mat * correct_ind_mat

    filter_correct_mat = list(map(lambda x, y: _compare(x, y), correct_pred_prob_mat, threshold_mat)) * np.ones(shape=correct_pred_prob_mat.shape)
    filter_correct_mat[filter_correct_mat > 0] = 1
    filter_correct_mat = np.sum(filter_correct_mat, axis=1)

    return filter_correct_mat


def cal_sub_corr_matrix(model, corr_path, submodels_path, fail_xs, fail_ys, fail_ys_label, fail_num, num_submodels=20):
    # add threshold
    sub_correct_matrix = None

    for root, dirs, files in os.walk(submodels_path):
        for i in range(num_submodels):
            temp_w_path = os.path.join(root, 'sub_{}.h5'.format(i))
            model.load_weights(temp_w_path)

            class_prob_mat = get_class_prob_mat(model, fail_xs, fail_ys)  # threshold

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
