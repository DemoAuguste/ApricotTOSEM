"""
realized functions:
* load_dataset(dataset='cifar10', preprocessing=True, shuffle=True)
* build_networks(model_name, num_classes, input_size)
* 

"""
import os
import settings
import numpy as np
from keras.datasets import cifar10, cifar100, mnist
import keras
from datetime import datetime
import settings
from keras.layers import Input
from sklearn.model_selection import train_test_split
from model import *

def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

def load_dataset(dataset='cifar10', preprocessing=True, shuffle=True):
    """
    return: x_train, x_test, y_train, y_test
    """
    if dataset == 'cifar10':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        if preprocessing:
            x_train, x_test = color_preprocessing(x_train, x_test)
        else:
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
    elif dataset == 'cifar100':
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif dataset == 'imagenet':  # imagenet64 dataset
        num_classes = 1000
        x_train = None
        y_train = None
        x_test = None
        y_test = None

        # load training dataset
        for i in range(10):
            temp_file_path = os.path.join(IMAGENET_DATASET_DIR, 'train_data_batch_{}'.format(i+1))
            temp_data = np.load(temp_file_path)
            temp = temp_data['data'].reshape(-1, 3, 64, 64)
            temp_x = np.rollaxis(temp, 1, 4)

            temp_y = np.array(temp_data['labels']) - 1
            temp_y = keras.utils.to_categorical(temp_y, num_classes)

            if x_train is None:
                x_train = temp_x
                y_train = temp_y
            else:
                x_train = np.concatenate(x_train, temp_x)
                y_train = np.concatenate(y_train, temp_y)

        # load test dataset
        test_file_path = os.path.join(IMAGENET_DATASET_DIR, 'val_data')
        test_data = np.load(test_file_path)

        x_test = test_data['data'].reshape(-1, 3, 64, 64)
        x_test = np.rollaxis(x_test, 1, 4)

        y_test = np.array(test_data['labels']) -1
        y_test = keras.utils.to_categorical(y_test, num_classes)

    # shuffle training dataset
    if shuffle:
        np.random.seed(settings.RANDOM_SEED)
        np.random.shuffle(x_train)
        np.random.seed(settings.RANDOM_SEED)
        np.random.shuffle(y_train)

    return x_train, x_test, y_train, y_test


def split_validation_dataset(xs, ys, val_rate=settings.VAL_RATE, random_seed=settings.RANDOM_SEED):
    x_train_val, x_val, y_train_val, y_val = train_test_split(xs, ys, test_size=val_rate, random_state=random_seed)
    return x_train_val, x_val, y_train_val, y_val


def build_networks(model_name, num_classes, input_size):
    input_tensor = Input(shape=input_size)
    top_k = 1 # default: only use top-1 accuracy.
    if model_name == 'resnet20':
        model = build_resnet(input_size[0], input_size[1], input_size[2], num_classes=num_classes, stack_n=3, k=top_k)
    elif model_name == 'resnet32':
        model = build_resnet(input_size[0], input_size[1], input_size[2], num_classes=num_classes, stack_n=5, k=top_k)
    elif model_name == 'mobilenet':
        model = build_mobilenet(input_tensor, num_classes, k=top_k)
    elif model_name == 'mobilenet_v2':
        model = build_mobilenet_v2(input_tensor, num_classes, k=top_k)
    elif model_name == 'densenet':
        model = build_densenet(input_tensor, num_classes, k=top_k)
    return model

def logger(msg, path):
    print(msg)
    now = datetime.now()
    str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    f = open(path, 'w+')
    w_msg = "[{}] {}\n".format(str_now, msg)
    f.write(w_msg)
    f.close()


def load_submodels(path, num_submodels):
    pass # no need to load models (consuming too many memories)


def get_submodels_weights(model, path):
    weights_list = []
    for root, dirs, files in os.walk(path):  # os.path.join(original_dir_path, 'submodels')
        for name in files:
            file_path = os.path.join(root, name)
            model.load_weights(file_path)
            weights_list.append(model.get_weights())
    return weights_list
        

