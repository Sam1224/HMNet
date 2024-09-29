#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset=pascal
exp_name=split$1  # 0
arch=HMNet
net=$2  # vgg/resnet50
postfix=$3  # manet/manet_5s

config=config/${dataset}/${dataset}_${exp_name}_${net}_${postfix}.yaml

python test_pascal.py --config=${config} --arch=${arch} --eposide=1000