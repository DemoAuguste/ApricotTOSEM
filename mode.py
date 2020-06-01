import tensorflow as tf
import keras as k
import numpy as np
# import matplotlib.pyplot as plt

# personal modules
from MODE.distance_metrics import bhattacharyya
from MODE.eval import get_eval_datasets, replicate_model
from MODE.layer_selection import (Forward_Layer_Select, 
                                calculate_acc_for_labels, 
                                get_overfit_labels, 
                                get_underfit_labels,
                                get_faultiest_label)
from MODE.input_selection import select_next_inputs
from MODE.heatmap import get_heatmaps

from utils import load_dataset, split_validation_dataset

from Apricot import *
from model import *
from utils import *
import argparse
import copy
from datetime import datetime


# Underfitting Threshold
# Both the training accuracy (TrAcc)  and the testing accuracy (TeAcc)
# must be lower than this param to qualify as underfitting
theta = 0.92

# Overfitting Threshold
# The difference between the training accuracy (TrAcc) and testing accuracy (TeAcc)
# must be larger than this param to qualify as overfitting
gamma = 0.10

# Ratio between Selected Data and Random Data 
# Controls the ratio of the target class to fix and the random other classes 
# trained simultaneosly. 
# TODO: find a way to experiment with setting this automatically based on the 
# number of classes in your task. 
# TODO: see if we can use larger alpha and reduced batch sizes to control overfitting
alpha = 0.25

# Byattacharyya Distance
# If the distribution of two y_pred matrices are less than this amount, than they are
# sufficiently similar and we'll take the layer that produced the distribution first
similarity_threshold = 0.01

# Number of Epochs
# TODO: implement early stopping
epochs = 3

# Batch Size
batch_size = 2000 # they used 2000 and 4000 in the paper







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
    # model.summary()
    model.load_weights(os.path.join(model_weights_dir, 'trained.h5'))

    log_path = os.path.join(model_weights_dir, 'mode.txt')

    model = replicate_model(model)
    model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # load dataset
    x_train, x_test, y_train, y_test = load_dataset(dataset)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_train, x_bug_fixes, y_train, y_bug_fixes = split_validation_dataset(x_train, y_train)

    # predict train (to get distributions)
    y_pred_train = model.predict(x_train)

    # predict test (to get distributions)
    y_pred = model.predict(x_test)


    ######################################
    max_iter = 5

    distance_metrics = ['dot']
    for distance_metric in distance_metrics:
        print('#####################################################################')
        print('Testing: {} similarity'.format(distance_metric))
        print('#####################################################################')

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print('Initial Test Loss: {}, Initial Test Accuracy: {}'.format(test_loss, test_acc))

        control_model = replicate_model(model)

        control_model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        # print('Control model performance to beat:')  
        # control_loss, control_acc = control_model.evaluate(x_test, y_test, verbose=0)
        # print('Control Test Loss: {}, Control Test Accuracy: {}'.format(control_loss, control_acc))
        # # control_model = copy.deepcopy(model)


        i = 0
        while i < max_iter:

            # control model with no specialized batch selection
            control_batch_x, control_batch_y = x_bug_fixes[:batch_size], y_bug_fixes[:batch_size]
            control_model.fit(control_batch_x, control_batch_y, epochs=epochs, verbose=0)

            control_loss, control_acc = control_model.evaluate(x_test, y_test, verbose=0)
            print('Control Test Loss: {}, Control Test Accuracy: {}'.format(control_loss, control_acc))

            logger(log_path, 'Control Test Loss: {}, Control Test Accuracy: {}'.format(control_loss, control_acc))

            # identify target layer
            print('Identifying target layer...')
            target_layer, cache = Forward_Layer_Select(model, 
                                                    x_train, 
                                                    y_train, 
                                                    x_test, 
                                                    y_test, 
                                                    epochs, 
                                                    similarity_threshold, 
                                                    verbose=True)

            feature_models, accuracies, bhattacharyyas, layer_pred = cache

            # breakout eval sets
            out = get_eval_datasets(data = x_test, 
                                    labels = y_test, 
                                    predictions = layer_pred)

            data_correct, labels_correct, data_incorrect, labels_incorrect, labels_corrected = out

            # generate heatmaps
            labels, heatmaps = get_heatmaps(correct_data = data_correct,
                                            correct_labels = labels_correct,
                                            misclassified_data = data_incorrect, 
                                            misclassified_labels = labels_incorrect, 
                                            misclassified_correct_labels = labels_corrected, 
                                            num_classes = num_classes,
                                            labels = [], 
                                            type = 'hci')

            # identify buggy labels
            accuracies_per_label_target = calculate_acc_for_labels(all_labels = y_test,
                                                                correct_labels = labels_correct,
                                                                num_classes = num_classes,
                                                                labels = [])

            underfit_labels = get_underfit_labels(acc = accuracies_per_label_target,
                                                num_classes = num_classes,
                                                threshold = theta,
                                                labels = []) 


            # select the most faulty underfitting label
            if len(underfit_labels) > 0:
                faultiest_label = get_faultiest_label(underfit_labels)
                print('Creating a batch for faultiest label: {}. There are {} faulty labels in total...'.format(faultiest_label, 
                                                                                                            len(underfit_labels)))
                print('Faulty labels: ', underfit_labels)
            else:
                print('MODE retraining complete!')
                break

            # generate a batch tailored to improving performance on that label
            next_X, next_y, x_bug_fixes, y_bug_fixes = select_next_inputs(bug_fix_data = x_bug_fixes,
                                                                        bug_fix_labels = y_bug_fixes,
                                                                        heatmaps = heatmaps,
                                                                        target_label = faultiest_label,
                                                                        batch_size = batch_size,
                                                                        ratio = alpha,
                                                                        for_underfitting = True,
                                                                        distance_metric = distance_metric)

            # train on this new batch
            model.fit(next_X, next_y, epochs=epochs, verbose=1)

            # evaluate performance
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

            logger(log_path, '{} iteration: test loss: {}, test accuracy: {}'.format(max_iter - i, test_loss, test_acc))

            # repeat until there are no more faulty underfitting labels
            i += 1

            print('{} iterations left...\n'.format(max_iter - i))

        print('Control model performance to beat:')  
        control_loss, control_acc = control_model.evaluate(x_test, y_test, verbose=0)
        print('Control Test Loss: {}, Control Test Accuracy: {}'.format(control_loss, control_acc))

        logger(log_path, 'Control Test Loss: {}, Control Test Accuracy: {}'.format(control_loss, control_acc))

