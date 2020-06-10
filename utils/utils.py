"""
realized functions:
* load_dataset(dataset='cifar10', preprocessing=True, shuffle=True)
* build_networks(model_name, num_classes, input_size)
* 

"""
import os
import settings
import numpy as np
from keras.datasets import cifar10, cifar100, mnist, imdb
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

def logger(path, msg):
    print(msg)
    str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    f = open(path, 'a+')
    w_msg = "[{}] {}\n".format(str_now, msg)
    f.write(w_msg)
    f.close()



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

    elif dataset == 'treebank':
        # f = open(os.path.join(DATA_DIR, "sentence_labels.txt"))
        # # text = f.read().decode("utf-8")
        # sentences= []
        # for sentence in f:
        #     sentence = sentence.split("\t")[0]
        #     sentences.append(sentence)
        # # sentences = sent_tokenize(text)
        # sentences = preprocess(sentences)
        # max_len = max_length(sentences)
        # word_model = gensim.models.Word2Vec(sentences, min_count=1)
        # f.close()
        pass

        X_train, y_train = get_file_data(os.path.join(DATA_DIR, 'train.txt'), word_model, max_len, word2vec_len, rnn_num_classes)
        X_valid, y_valid = get_file_data(os.path.join(DATA_DIR, "valid.txt"), word_model, max_len, word2vec_len, rnn_num_classes)
        X_test, y_test = get_file_data(os.path.join(DATA_DIR, "test.txt"), word_model, max_len, word2vec_len, rnn_num_classes)

        return X_train, y_train, X_valid, y_valid, X_test, y_test, max_len
    
    elif dataset == 'imdb':
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
        x_train = sequence.pad_sequences(x_train, maxlen=100)
        x_test = sequence.pad_sequences(x_test, maxlen=100)

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
    top_k = 1 # default: only use top-1 accuracy.
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
    elif model_name == 'lstm':
        max_features = 20000
        model = lstm(max_features)
    elif model_name == 'bilstm':
        max_features = 20000
        model = bilistm(max_features)
    return model

def logger(path, msg):
    print(msg)
    now = datetime.now()
    str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    f = open(path, 'a+')
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
        

# RNN use
def get_file_data(filename, model, max_len, word2vec_len, num_classes):
	sentences = []
	labels = []
	sentence_labels = open(filename)
	for sentence_label in sentence_labels:
		sentence = sentence_label.split("\t")[0]
		sentences.append(sentence)
		label = sentence_label.split("\t")[2]
		labels.append(int(label))
	sentences = preprocess(sentences)
	sentence_labels.close

	X = np.zeros((len(sentences),max_len, word2vec_len))
	y = np.zeros((len(sentences),num_classes))

	for i in range(len(sentences)):
		for j in range(len(sentences[i])):
			X[i,j,:] = model[sentences[i][j]]
		# print labels[i],
		y[i,labels[i]]=1

	return X,y

def preprocess(sentences):
	# for i in range(len(sentences)):
	# 	sentences[i] = word_tokenize(sentences[i])
		# sentences[i] = [word for word in sentences[i] if word not in stop_words]
		# sentences[i] = [stemmer.stem(word) for word in sentences[i]]
	return sentences

def max_length(sentences):
	max_len=-1
	for i in range(len(sentences)):
		if len(sentences[i])>max_len:
			max_len=len(sentences[i])
	return max_len
