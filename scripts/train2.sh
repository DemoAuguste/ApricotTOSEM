#!/bin/bash

# initialization
working_dir=$(dirname "$PWD")
script_name="main_train.py"
# echo $current_path
# echo $working_dir

train_path=$working_dir"/"$script_name

echo $train_path

# script for training models

# svhn
for i in {11..15}
do
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m resnet20 -d svhn -v $i -pre 2 -after 48 -st 18
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m mobilenet -d svhn -v $i -pre 2 -after 48 -st 18
    CUDA_VISIBLE_DEVICES=2 python3 $2rain_path -m resnet32 -d svhn -v $i -pre 2 -after 48 -st 18
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m mobilenetv2 -d svhn -v $i -pre 2 -after 48 -st 18
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m densenet -d svhn -v $i -pre 2 -after 48 -st 18
done

# fashion-mnist
for i in {11..15}
do
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m resnet20 -d fashion-mnist -v $i -pre 1 -after 49 -st 9
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m mobilenet -d fashion-mnist -v $i -pre 1 -after 49 -st 9
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m resnet32 -d fashion-mnist -v $i -pre 1 -after 49 -st 9
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m mobilenetv2 -d fashion-mnist -v $i -pre 1 -after 49 -st 9
    CUDA_VISIBLE_DEVICES=2 python3 $train_path -m densenet -d fashion-mnist -v $i -pre 1 -after 49 -st 9
done

