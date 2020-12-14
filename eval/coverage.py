import sys

sys.path.append('..')
from model import *
from utils import *
import argparse
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model to be fixed.', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='number of version.', type=int)

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version

    # initialization.
    num_classes = 10
    input_size = (32, 32, 3)
    if dataset == 'cifar10':
        num_classes = 10
        input_size = (32, 32, 3)
    elif dataset == 'cifar100':
        num_classes = 100
        input_size = (32, 32, 3)
    elif dataset == 'svhn':
        num_classes = 10
        input_size = (32, 32, 3)
    elif dataset == 'fashion-mnist' or dataset == 'mnist':
        num_classes = 10
        input_size = (28, 28, 1)
    else:
        NotImplementedError('Not implemented.')  # TODO: extend other dataset

    model = build_networks(model_name, num_classes, input_size)
    model_weights_dir = os.path.join(WORKING_DIR, 'weights')
    model_weights_dir = os.path.join(model_weights_dir, model_name)
    model_weights_dir = os.path.join(model_weights_dir, dataset)
    model_weights_dir = os.path.join(model_weights_dir, str(ver))
    # model.summary()
    # model.load_weights(os.path.join(model_weights_dir, 'trained.h5'))
    sub_dir = os.path.join(model_weights_dir, 'submodels')

    x_train, x_test, y_train, y_test = load_dataset(dataset, shuffle=False)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    sum_vec = None
    y_label = np.argmax(y_train_val, axis=1)

    for i in range(20):
        temp_path = os.path.join(sub_dir, 'sub_{}.h5'.format(i))
        model.load_weights(temp_path)
        # print(model.evaluate(x_train_val, y_train_val))
        preds = model.predict(x_train_val)
        preds_label = np.argmax(preds, axis=1)
        temp_ind = preds_label == y_label
        print(np.sum(temp_ind))
        # temp_ind = np.array(temp_ind, dtype=np.int)
        if sum_vec is None:
            sum_vec = copy.deepcopy(temp_ind)
        else:
            sum_vec = sum_vec | temp_ind
    # sum_vec[sum_vec > 0] = 1
    print('submodel train data coverage: {} / {} = {:.4f}'.format(np.sum(sum_vec), x_train_val.shape[0],
                                                                  np.sum(sum_vec) / x_train_val.shape[0]))

    # original model prediction
    trained_path = os.path.join(model_weights_dir, 'trained.h5')
    model.load_weights(trained_path)
    ret = model.predict(x_train_val)
    temp = np.argmax(ret, axis=1)
    trained_pred_label = temp == y_label
    xor = trained_pred_label ^ sum_vec
    # rDLM correct part
    sub_correct = sum_vec & xor
    # original correct part
    origin_correct = trained_pred_label & xor
    print('submodel correct part: {} / {} = {:.4f}'.format(np.sum(sub_correct), x_train_val.shape[0],
                                                           np.sum(sub_correct) / x_train_val.shape[0]))
    print('original correct part: {} / {} = {:.4f}'.format(np.sum(origin_correct), x_train_val.shape[0],
                                                           np.sum(origin_correct) / x_train_val.shape[0]))



