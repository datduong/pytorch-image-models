#!/bin/bash


data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/
output=/data/duongdb/HAM10000dataset/inception_v3/
batchsize=128


cd /data/duongdb/pytorch-image-models


# add weighted cross entropy 

# https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/inceptionv3-mixup.sh
# python train_imagenet.py \
#   --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
#   --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
#   --model inceptionv3 --mode hybrid --input-size 299 \
#   --lr 0.4 --lr-mode cosine --num-epochs 200 --batch-size 128 --num-gpus 8 -j 60 \
#   --use-rec --dtype float16 --warmup-epochs 5 --no-wd --label-smoothing --mixup \
#   --save-dir params_inceptionv3_mixup \
#   --logging-file inceptionv3_mixup.log


CUDA_VISIBLE_DEVICES=0,2 python3 train.py $data_path --model inception_v3 -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 --pretrained --num-classes 7 --output $output --weighted_cross_entropy '30.62691 19.48443 9.11282 87.08695 8.99820 1.49366 70.52816'



