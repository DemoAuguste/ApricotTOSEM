from sklearn.metrics import classification_report
import numpy as np

def get_class_acc(model, xs, ys):
    y_pred = model.predict(xs)
    print(y_pred.shape)

    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0

    print(classification_report(ys, y_pred, output_dict=True))
    return classification_report(ys, y_pred, output_dict=True)


def get_class_prob_mat(model, xs, ys):
    pred_prob_mat = model.predict(xs)

    max_ind_mat = list(map(lambda x: x==max(x), pred_prob_mat)) * np.ones(shape=pred_prob_mat.shape)
    max_prob_mat = pred_prob_mat * max_ind_mat * ys
    sum_prob_mat = np.sum(max_prob_mat, axis=0)
    class_sum_mat = np.sum(max_ind_mat, axis=0)
    class_prob_mat = sum_prob_mat / class_sum_mat
    
    print(class_prob_mat)
    return class_prob_mat

    # print(np.sum(max_ind_mat * ys) / len(xs))
    # print(model.evaluate(x_train, y_train))

def _compare(x,y):
    if x > y:
        return x
    else:
        return 0


def get_model_correct_mat(model, xs, ys):
    pred_prob_mat = model.predict(xs)
    class_prob_mat = get_class_prob_mat(model, xs, ys) # threshold

    max_ind_mat = list(map(lambda x: x==max(x), pred_prob_mat)) * np.ones(shape=pred_prob_mat.shape)
    correct_ind_mat = max_ind_mat * ys # if model predicts correctly, then one row should have one element equals to 1
    correct_pred_prob_mat = pred_prob_mat * correct_ind_mat

    # construct a threshold matrix
    threshold_mat = np.tile(class_prob_mat, (len(xs),1))
    threshold_mat = threshold_mat * correct_ind_mat

    filter_correct_mat = list(map(lambda x, y: _compare(x, y), correct_pred_prob_mat, threshold_mat)) * np.ones(shape=correct_pred_prob_mat.shape)
    filter_correct_mat[filter_correct_mat > 0] = 1
    filter_correct_mat = np.sum(filter_correct_mat, axis=1)

    return filter_correct_mat
    
