#!/bin/bash

# initialization
working_dir=$(dirname "$PWD")
script_name="apricot_main.py"
# echo $current_path
# echo $working_dir

file_path=$working_dir"/"$script_name

echo $file_path

# CIFAR-100 strategy 2
for i in {11..15}
do
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet20 -d cifar100 -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet32 -d cifar100 -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenet -d cifar100 -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenetv2 -d cifar100 -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m densenet -d cifar100 -v $i -s 2
done

# SVHN strategy 2
for i in {11..15}
do
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet20 -d svhn -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet32 -d svhn -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenet -d svhn -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenetv2 -d svhn -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m densenet -d svhn -v $i -s 2
done

# fashion-mnist strategy 2
for i in {11..15}
do
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet20 -d fashion-mnist -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m resnet32 -d fashion-mnist -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenet -d fashion-mnist -v $i -s 2
  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m mobilenetv2 -d fashion-mnist -v $i -s 2
#  CUDA_VISIBLE_DEVICES=2 python3 $file_path -m densenet -d fashion-mnist -v $i -s 2  # NOT TRAINED!
done

