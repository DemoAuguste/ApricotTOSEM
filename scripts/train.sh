#!/bin/bash

# initialization
working_dir=$(dirname "$PWD") 
script_name="main_train.py"
# echo $current_path
# echo $working_dir

train_path=$working_dir"/"$script_name

echo $train_path

# script for training models
for i in {2..5}
do
    CUDA_VISIDBLE_DEVICES=0 python $train_path -m resnet20 -d cifar100 -v $k
done