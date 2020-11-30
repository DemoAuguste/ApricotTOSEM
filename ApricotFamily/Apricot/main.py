# the main process of Apricot (conference version).
import keras
import os
import numpy as np
from utils import load_dataset, split_validation_dataset

def apricot(model, model_weights_dir, adjustment_strategy):
    x_train, x_test, y_train, y_test = load_dataset(dataset)
    x_train_val, x_val, y_train_val, y_val = split_validation_dataset(x_train, y_train)

    fixed_model = model
    submodel_dir = os.path.join(model_weights_dir, 'submodels')
    trained_weights_path = os.path.join(model_weights_dir, 'trained.h5')
    fixed_weights_path = os.path.join(model_weights_dir, 'fixed_{}.h5'.format(adjustment_strategy))
    log_path = os.path.join(model_weights_dir, 'log_{}.h5')
