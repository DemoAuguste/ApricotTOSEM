from utils import *
# from Apricot import *
from model import *

import argparse


def training_process(arguments):
    # CIFAR-10 training process.
    num_classes = None
    if arguments.dataset == 'cifar10' or arguments.dataset == 'svhn' or arguments.dataset == 'fashion-mnist':
        num_classes = 10
    elif arguments.dataset == 'cifar100':
        num_classes = 100
    train_sub = True
    if arguments.dataset != 'imdb':
        train_model(arguments.model,
                    num_classes=num_classes,
                    dataset=arguments.dataset,
                    ver=arguments.version,
                    num_submodels=20,
                    train_sub=train_sub,
                    pre_epochs=arguments.pre_epochs,
                    after_epochs=arguments.after_epochs,
                    sub_epochs=arguments.sub_epochs)
    else:
        NotImplementedError('Not implemented.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='version number.', type=int, default=99)
    parser.add_argument('-s', '--submodels', help='number of submodels', type=int, default=20)
    parser.add_argument('-pre', '--pre_epochs', help='number of epochs for pretraining the model', type=int, default=10)
    parser.add_argument('-after', '--after_epochs', help='number of epochs for training the model', type=int, default=190)
    parser.add_argument('-st', '--sub_epochs', help='number of epochs for training the submodel', type=int, default=40)
    args = parser.parse_args()

    training_process(args)
