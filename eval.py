from Apricot import *
from model import *
from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model to be fixed.', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='number of version.', type=int)
    parser.add_argument('-n', '--name', help='the name of the file', type=str, default='trained.h5')

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version
    f_name = args.name

    # initialization.
    if dataset == 'cifar10':
        num_classes = 10
        input_size = (32, 32, 3)
    elif dataset == 'cifar100':
        num_classes = 100
        input_size = (32, 32, 3)
    else:
        pass # TODO: extend other dataset

    model = build_networks(model_name, num_classes, input_size)
    model_weights_dir = os.path.join(WORKING_DIR, 'weights')
    model_weights_dir = os.path.join(model_weights_dir, model_name)
    model_weights_dir = os.path.join(model_weights_dir, dataset)
    model_weights_dir = os.path.join(model_weights_dir, str(ver))
    # model.summary()
    model.load_weights(os.path.join(model_weights_dir, f_name))

    # evaluation.
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    _, val_acc = fixed_model.evaluate(x_train_val, y_train_val)
    print('training acc: {:.4f}'.format(val_acc))
    _, val_acc = fixed_model.evaluate(x_val, y_val)
    print('validation acc: {:.4f}'.format(val_acc))
    _, val_acc = fixed_model.evaluate(x_test, y_test)
    print('test acc: {:.4f}'.format(val_acc))