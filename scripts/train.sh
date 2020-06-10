#!/bin/bash

# initialization
working_dir=$(dirname "$PWD") 
script_name="main_train.py"
# echo $current_path
# echo $working_dir

train_path=$working_dir"/"$script_name

echo $train_path

# script for training models

# CIFAR-100
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m resnet20 -d cifar100 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m mobilenet -d cifar100 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m resnet32 -d cifar100 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m mobilenetv2 -d cifar100 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m densenet -d cifar100 -v $i
done

# CIFAR-10 
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m resnet20 -d cifar10 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m mobilenet -d cifar10 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m resnet32 -d cifar10 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m mobilenetv2 -d cifar10 -v $i
    CUDA_VISIBLE_DEVICES=3 python3 $train_path -m densenet -d cifar10 -v $i
done

