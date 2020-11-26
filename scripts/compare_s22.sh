#!/bin/bash

# initialization
working_dir=$(dirname "$PWD") 
script_name="main_adaptation3.py"
# echo $current_path
# echo $working_dir

adaptation_path=$working_dir"/"$script_name

echo "$adaptation_path"



for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=3 python3 "$adaptation_path" -m resnet20 -d cifar10 -v $i -s 2 -a binary
    CUDA_VISIBLE_DEVICES=3 python3 "$adaptation_path" -m mobilenet -d cifar10 -v $i -s 2 -a binary
    CUDA_VISIBLE_DEVICES=3 python3 "$adaptation_path" -m resnet32 -d cifar10 -v $i -s 2 -a binary
    CUDA_VISIBLE_DEVICES=3 python3 "$adaptation_path" -m mobilenetv2 -d cifar10 -v $i -s 2 -a binary
    CUDA_VISIBLE_DEVICES=3 python3 "$adaptation_path" -m densenet -d cifar10 -v $i -s 2 -a binary
done
