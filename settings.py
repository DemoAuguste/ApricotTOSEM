"""
--------------
ver
--------------
start with 1: train on CIFAR-10
start with 2: train on CIFAR-100
start with 3: train on ImageNet-64
"""


import os, sys
import random

WORKING_DIR = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(WORKING_DIR, 'weights')

if 'win' in sys.platform:
    IMAGENET_DATASET_DIR = 'D:\\imagenet'
else:
    IMAGENET_DATASET_DIR = '/home/grads/hzhang339/imagenet'

LOGGER_DIR = os.path.join(WORKING_DIR, 'log')

# hyperparameters
learning_rate = 0.01

# strategy = 1

# train DL model
# BATCH_SIZE          = 128
# EPOCHS              = 50
# ITERATIONS      = 50000 // BATCH_SIZE + 1
# NUM_REDUCED_MODEL   = 20

"""
The training parameters should be different when training model on ImageNet dataset
"""
BATCH_SIZE = 128

PRE_EPOCHS = 10
AFTER_EPOCHS = 190
SUB_EPOCHS = 40  # 20% of pre + after
MAX_COUNT = 50
LOOP_COUNT = 100

FURTHER_ADJUSTMENT_EPOCHS = 20

# parameters for validation dataset
RANDOM_SEED = 42
VAL_RATE = 0.2

MONITOR = 'val_acc'
