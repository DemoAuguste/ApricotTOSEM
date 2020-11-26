"""
realized functions:
* load_dataset(dataset='cifar10', preprocessing=True, shuffle=True)
* build_networks(model_name, num_classes, input_size)
* 

"""
import os
import settings
import numpy as np
from keras.datasets import cifar10, cifar100, fashion_mnist
import scipy.io as spio
import keras
from datetime import datetime
from keras.layers import Input
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import *
# import gensim
from settings import *

from model import *
from datetime import datetime


def print_msg(msg):
    print(msg)


# region SVHN dataset processing functions
# ---- Function to convert rgb images to grayscale -----#
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# ---- Function to format data to use with keras ----#
def formatArray(data):
    im = []
    for i in range(0, data.shape[3]):
        # im.append(rgb2gray(data[:, :, :, i]))
        im.append(data[:, :, :, i])
    return np.asarray(im)


# ---- Replace 10 in labels with 0 ----#
def fixLabel(labels):
    labels[labels == 10] = 0
    return labels
# endregion


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
    x_train = None
    y_train = None
    x_test = None
    y_test = None
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

        # load test dataset
        test_file_path = os.path.join(IMAGENET_DATASET_DIR, 'val_data')
        test_data = np.load(test_file_path)

        x_test = test_data['data'].reshape(-1, 3, 64, 64)
        x_test = np.rollaxis(x_test, 1, 4)

        y_test = np.array(test_data['labels']) -1
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif dataset == 'svhn':
        # code from https://github.com/haseebtehsin/Neural-Network-using-Tensorflow-keras-and-SVHN-Dataset/blob/master/NN.py
        data_path = os.path.join(DATA_DIR, 'svhn')
        mat1 = spio.loadmat(os.path.join(data_path, 'train_32x32.mat'), squeeze_me=True)
        mat2 = spio.loadmat(os.path.join(data_path, 'test_32x32.mat'), squeeze_me=True)
        x_train = mat1['X']
        y_train = mat1['y']
        x_test = mat2['X']
        y_test = mat2['y']
        # ------------- Convert to proper format -------------#
        x_train = formatArray(x_train)
        x_test = formatArray(x_test)
        y_train = fixLabel(y_train)
        y_test = fixLabel(y_test)
        # ------------- Normalize ---------------#
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0

    elif dataset == 'fashion-mnist':
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

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


def build_networks(model_name, num_classes=None, input_size=None):
    # input_tensor = Input(shape=input_size)
    top_k = 1  # default: only use top-1 accuracy.
    model = None
    if model_name == 'resnet20':
        input_tensor = Input(shape=input_size)
        model = build_resnet(input_size[0], input_size[1], input_size[2], num_classes=num_classes, stack_n=3, k=top_k)
    elif model_name == 'resnet32':
        input_tensor = Input(shape=input_size)
        model = build_resnet(input_size[0], input_size[1], input_size[2], num_classes=num_classes, stack_n=5, k=top_k)
    elif model_name == 'mobilenet':
        input_tensor = Input(shape=input_size)
        model = build_mobilenet(input_tensor, num_classes, k=top_k)
    elif model_name == 'mobilenetv2':
        input_tensor = Input(shape=input_size)
        model = build_mobilenet_v2(input_tensor, num_classes, k=top_k)
    elif model_name == 'densenet':
        input_tensor = Input(shape=input_size)
        model = build_densenet(input_tensor, num_classes, k=top_k)
    # elif model_name == 'lstm':
    #     max_features = 20000
    #     model = lstm(max_features)
    # elif model_name == 'bilstm':
    #     max_features = 20000
    #     model = bilistm(max_features)
    return model


def logger(msg, path):
    print(msg)
    now = datetime.now()
    str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    f = open(path, 'a+')
    w_msg = "[{}] {}\n".format(str_now, msg)
    f.write(w_msg)
    f.close()


def load_submodels(path, num_submodels):
    pass # no need to load models (consuming too many memories)


def get_submodels_weights(model, path, num_submodels=20):
    weights_list = []
    for root, dirs, files in os.walk(path):  # os.path.join(original_dir_path, 'submodels')
        for i in range(num_submodels):
            file_path = os.path.join(root, 'sub_{}.h5'.format(i))
            model.load_weights(file_path)
            weights_list.append(model.get_weights())
    return weights_list
        

# RNN use
# def get_file_data(filename, model, max_len, word2vec_len, num_classes):
# 	sentences = []
# 	labels = []
# 	sentence_labels = open(filename)
# 	for sentence_label in sentence_labels:
# 		sentence = sentence_label.split("\t")[0]
# 		sentences.append(sentence)
# 		label = sentence_label.split("\t")[2]
# 		labels.append(int(label))
# 	sentences = preprocess(sentences)
# 	sentence_labels.close
#
# 	X = np.zeros((len(sentences),max_len, word2vec_len))
# 	y = np.zeros((len(sentences),num_classes))
#
# 	for i in range(len(sentences)):
# 		for j in range(len(sentences[i])):
# 			X[i,j,:] = model[sentences[i][j]]
# 		# print labels[i],
# 		y[i,labels[i]]=1
#
# 	return X,y
#
# def preprocess(sentences):
# 	# for i in range(len(sentences)):
# 	# 	sentences[i] = word_tokenize(sentences[i])
# 		# sentences[i] = [word for word in sentences[i] if word not in stop_words]
# 		# sentences[i] = [stemmer.stem(word) for word in sentences[i]]
# 	return sentences
#
# def max_length(sentences):
# 	max_len=-1
# 	for i in range(len(sentences)):
# 		if len(sentences[i])>max_len:
# 			max_len=len(sentences[i])
# 	return max_len
