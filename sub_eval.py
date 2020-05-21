from Apricot import *
from model import *
from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model to be fixed.', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='number of version.', type=int)
    parser.add_argument('-n', '--num', help='the name of the file', type=int, default=20)

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version
    f_name = args.name
    num = args.num

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
    model_weights_dir = os.path.join(model_weights_dir, 'submodels')
    # model.summary()
    # model.load_weights(os.path.join(model_weights_dir, f_name))

    # evaluation.
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    total_train_acc = 0.0
    total_val_acc = 0.0
    total_test_acc = 0.0

    for i in range(num):
        sub_path = os.path.join(model_weights_dir)
        model.load_weights(sub_path)

        _, val_acc = model.evaluate(x_train_val, y_train_val)
        print('training acc: {:.4f}'.format(val_acc))
        total_train_acc += val_acc

        _, val_acc = model.evaluate(x_val, y_val)
        print('validation acc: {:.4f}'.format(val_acc))
        total_val_acc += val_acc

        _, val_acc = model.evaluate(x_test, y_test)
        print('test acc: {:.4f}'.format(val_acc))
        total_test_acc += val_acc

    print('avg train acc: {:.4f}, val acc: {:.4f}, test acc: {:4f}'.format(total_train_acc/num, total_val_acc/num, total_test_acc/num))