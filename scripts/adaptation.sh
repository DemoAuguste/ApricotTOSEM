#!/bin/bash

# initialization
working_dir=$(dirname "$PWD") 
script_name="main_adaptation.py"
# echo $current_path
# echo $working_dir

adaptation_path=$working_dir"/"$script_name

echo $train_path

for i in {2..5}
do
    CUDA_VISIBLE_DEVICES=1 python $adaptation_path -m resnet20 -d cifar100 -v $i -s 1 -a binary
    # CUDA_VISIBLE_DEVICES=0 python $train_path -m mobilenet -d cifar100 -v $i -s 1 -a binary
    # CUDA_VISIBLE_DEVICES=0 python $train_path -m resnet32 -d cifar100 -v $i -s 1 -a binary
    # CUDA_VISIBLE_DEVICES=0 python $train_path -m mobilenetv2 -d cifar100 -v $i -s 1 -a binary
done

for i in {1..3}
do
    CUDA_VISIBLE_DEVICES=1 python $train_path -m mobilenet -d cifar100 -v $i -s 1 -a binary
    CUDA_VISIBLE_DEVICES=1 python $train_path -m resnet32 -d cifar100 -v $i -s 1 -a binary
    # CUDA_VISIBLE_DEVICES=1 python $train_path -m mobilenetv2 -d cifar100 -v $i -s 1 -a binary
done