#!/bin/bash


data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/
output=/data/duongdb/HAM10000dataset/efficientnet_b0/
batchsize=64


cd /data/duongdb/pytorch-image-models


# add weighted cross entropy 

# --lr .048  

python3 train.py $data_path --model efficientnet_b0 -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .001 --pretrained --num-classes 7 --output $output --weighted_cross_entropy '30.62691 19.48443 9.11282 87.08695 8.99820 1.49366 70.52816' 

# --resume $output/train/20200819-114953-efficientnet_b0-224/model_best.pth.tar




