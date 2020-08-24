#!/bin/bash


data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/
output=/data/duongdb/HAM10000dataset/efficientnet_b2/
batchsize=64

cd /data/duongdb/pytorch-image-models


python3 train.py $data_path --model efficientnet_b2 -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 4 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr 0.001 --classification_layer_name 'classifier' --filter_bias_and_bn --pretrained --num-classes 7 --output $output --weighted_cross_entropy '30.62691 19.48443 9.11282 87.08695 8.99820 1.49366 70.52816' 


# --resume $output/train/20200820-003052-efficientnet_b2-260/model_best.pth.tar



