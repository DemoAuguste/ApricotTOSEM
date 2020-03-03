from Apricot import *
from model import *
from utils import *
import argparse


def adaptation_process():
    apricot(model, model_weights_dir, dataset, adjustment_strategy, activation='binary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model to be fixed.', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='number of version.', type=int)
    parser.add_argument('-s', '--strategy', help='adjustment strategy.', type=int)
    parser.add_argument('-a', '--activation', help='activation function.', type=str, default='binary')

    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    ver = args.version
    adjustment_strategy = args.strategy
    activation = args.activation

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
    

    # the main process of Apricot
    apricot(model, model_weights_dir, dataset, adjustment_strategy, activation)
    

    