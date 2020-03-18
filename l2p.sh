#! /bin/bash

PYTHON="/home/kaka/anaconda3/bin/python3"

####### Args ################
dataset=cifar10
baseline=resnet_v1
res_v1_depth=20
learning_rate=0.1
weight_decay=3e-4
model_base=cifar10_repo/

{
$PYTHON main.py --dataset ${dataset}  --baseline ${baseline} --res_v1_depth ${res_v1_depth} \
    --learning_rate ${learning_rate}  --weight_decay ${weight_decay}  --model_base ${model_base}
}