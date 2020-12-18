#!/bin/bash

# initialization
working_dir=$(dirname "$PWD")
script_name="apricorn_main.py"
# echo $current_path
# echo $working_dir

file_path=$working_dir"/"$script_name

echo $file_path

# CIFAR-100 strategy 2
#for i in {11..15}
#do
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet20 -d cifar100 -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet32 -d cifar100 -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenet -d cifar100 -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenetv2 -d cifar100 -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m densenet -d cifar100 -v $i
#done
#
## SVHN strategy 2
#for i in {11..15}
#do
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet20 -d svhn -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet32 -d svhn -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenet -d svhn -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenetv2 -d svhn -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m densenet -d svhn -v $i
#done

# fashion-mnist strategy 2
for i in {11..15}
do
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet20 -d fashion-mnist -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m resnet32 -d fashion-mnist -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenet -d fashion-mnist -v $i
#  CUDA_VISIBLE_DEVICES=1 python3 $file_path -m mobilenetv2 -d fashion-mnist -v $i
  CUDA_VISIBLE_DEVICES=3 python3 $file_path -m densenet -d fashion-mnist -v $i  # NOT TRAINED!
done

