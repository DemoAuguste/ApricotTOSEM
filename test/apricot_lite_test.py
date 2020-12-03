import sys
sys.path.append('..')
import argparse
from ApricotFamily.ApricotLite import *
from utils import *


def apricot_plus_process(args):
    img_rows = -1
    img_cols = -1
    img_channels = -1
    num_classes = 10
    if 'cifar' in args.dataset or args.dataset == 'svhn':
        img_rows, img_cols = 32, 32
        img_channels = 3
    elif args.dataset == 'fashion-mnist' or args.dataset == 'mnist':
        img_rows, img_cols = 28, 28
        img_channels = 1
    else:  # TODO
        pass
    if args.dataset == 'cifar100':
        num_classes = 100

    model_weights_save_dir = os.path.join(WORKING_DIR, 'weights')
    model_weights_save_dir = os.path.join(model_weights_save_dir, args.model)
    model_weights_save_dir = os.path.join(model_weights_save_dir, args.dataset)
    model_weights_save_dir = os.path.join(model_weights_save_dir, str(args.version))

    if not os.path.exists(model_weights_save_dir):
        os.makedirs(model_weights_save_dir)

    model = build_networks(args.model, num_classes=num_classes, input_size=(img_rows, img_cols, img_channels))

    apricot_lite(model, model_weights_save_dir, args.dataset, args.strategy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings of Apricot+.')
    parser.add_argument('-m', '--model', help='model', type=str, default='resnet20')
    parser.add_argument('-d', '--dataset', help='dataset for training the model.', type=str, default='cifar10')
    parser.add_argument('-v', '--version', help='version number.', type=int, default=99)
    parser.add_argument('-s', '--strategy', help='strategy', type=int,
                        default=2)  # chose 2 because 2 is the best in conference version.
    args = parser.parse_args()

    apricot_plus_process(args)
