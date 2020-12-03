from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from utils import load_dataset, split_validation_dataset
from utils import logger
from settings import NUM_SUBMODELS, LOOP_COUNT, FIX_BATCH_SIZE
from .func import *
from settings import BATCH_SIZE, FURTHER_ADJUSTMENT_EPOCHS
import os
import numpy as np
from .func import *
from ApricotFamily.Apricot.func import get_indexed_failing_cases, apricot_cal_sub_corr_mat, get_weights_list
from datetime import datetime


def apricot_lite(model, model_weights_dir, dataset, adjustment_strategy=None):
    """
    apricot lite
    does not need strategy.
    """
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    train_size = len(x_train)
    val_size = len(x_train_val)
    test_size = len(x_test)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'apricot_lite_fixed.h5')
    log_path = os.path.join(model_weights_dir, 'apricot_lite.log')

    if not os.path.join(fixed_weights_path):
        fixed_model.save_weights(fixed_weights_path)












