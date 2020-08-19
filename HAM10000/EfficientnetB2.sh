#!/bin/bash


data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/
output=/data/duongdb/HAM10000dataset/efficientnet_b2/
batchsize=64

cd /data/duongdb/pytorch-image-models

./distributed_train.sh 2 $data_path --model efficientnet_b2 -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --pretrained --num-classes 7 --output $output




