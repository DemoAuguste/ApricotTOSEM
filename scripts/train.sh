#!/bin/bash

# initialization
working_dir=$(dirname "$PWD") 
script_name="main_train.py"
# echo $current_path
# echo $working_dir

train_path=$working_dir"/"$script_name

echo $train_path
read


# script for training models
# CUDA_VISIDBLE_DEVICES=0,1 python 