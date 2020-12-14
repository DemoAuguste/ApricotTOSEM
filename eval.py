from Apricot import *
from model import *
from utils import *
import argparse

def cal_mean_and_std(input_list):
    print('mean: {:.4f}, std: {:.4f}'.format(float(np.mean(input_list)), float(np.std(input_list))))


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

    x_train, x_test, y_train, y_test = load_dataset(dataset)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    # evaluate submodels
    avg_train_acc = []
    avg_val_acc = []
    avg_test_acc = []
    for i in range(20):
        print('[sub {}] '.format(i))
        temp_path = os.path.join(sub_dir, 'sub_{}.h5'.format(i))
        model.load_weights(temp_path)
        ret = model.evaluate(x_train_val, y_train_val)
        avg_train_acc.append(float(ret[1]))
        ret = model.evaluate(x_val, y_val)
        avg_val_acc.append(float(ret[1]))
        ret = model.evaluate(x_test, y_test)
        avg_test_acc.append(float(ret[1]))

    # mean and std
    avg_train_acc = np.array(avg_train_acc).flatten()
    avg_val_acc = np.array(avg_val_acc).flatten()
    avg_test_acc = np.array(avg_test_acc).flatten()
    cal_mean_and_std(avg_train_acc)
    cal_mean_and_std(avg_val_acc)
    cal_mean_and_std(avg_test_acc)

    # apricorn
    print('------ fixed by apricorn -----')
    temp_path = os.path.join(model_weights_dir, 'apricorn_fixed.h5')
    model.load_weights(temp_path)
    print('train acc:')
    print(model.evaluate(x_train_val, y_train_val))
    print('validation acc:')
    print(model.evaluate(x_val, y_val))
    print('test acc:')
    print(model.evaluate(x_test, y_test))

    

