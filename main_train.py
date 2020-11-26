from utils import *
# from Apricot import *
from model import *

import argparse


def training_process(args):
    # cifar 10 training process.
    num_classes = None
    if args.dataset == 'cifar10' or args.dataset == 'svhn' or args.dataset == 'fashion-mnist':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    train_sub = True
    if args.dataset != 'imdb':
        train_model(args.model, num_classes=num_classes, dataset=args.dataset, ver=args.version, num_submodels=20,
                    train_sub=train_sub, save_path=None,
                    pre_epochs=args.pre_epochs, after_epochs=args.after_epochs, sub_epochs=args.sub_epochs)
    else:
        pass
        # rnn_train_model(model_name, num_classes=2, dataset=dataset, ver=ver, num_submodels=40, train_sub=train_sub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar100')
    parser.add_argument('-v', '--version', help='version number.', type=int, default=99)
    parser.add_argument('-s', '--submodels', help='number of submodels', type=int, default=20)
    parser.add_argument('-pre', '--pre_epochs', help='number of epochs for pretraining the model', type=int, default=10)
    parser.add_argument('-after', '--after_epochs', help='number of epochs for training the model', type=int, default=190)
    parser.add_argument('-st', '--sub_epochs', help='number of epochs for training the submodel', type=int,
                        default=40)
    args = parser.parse_args()

    training_process(args)
