from utils import *
# from Apricot import *
from model import *

import argparse

def training_process(model_name, dataset, ver):
    # cifar 10 training process.
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    train_sub = True
    if dataset != 'imdb':
        train_model(model_name, num_classes=num_classes, dataset=dataset, ver=ver, num_submodels=settings.NUM_SUBMODELS, train_sub=train_sub, save_path=None)
    else:
        rnn_train_model(model_name, num_classes=2, dataset=dataset, ver=ver, num_submodels=settings.NUM_SUBMODELS, train_sub=train_sub)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model', type=str)
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar100')
    parser.add_argument('-v','--version', help='version number.', type=int, default=2)
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    ver = args.version

    training_process(model, dataset, ver)
    