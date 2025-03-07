#!/bin/bash


data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/
model=efficientnet_b1
output=/data/duongdb/HAM10000dataset/$model/
batchsize=64

cd /data/duongdb/pytorch-image-models

python3 train.py $data_path --model $model -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 --pretrained --num-classes 7 --output $output 




