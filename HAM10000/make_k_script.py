
import os,sys,re,pickle,time
from datetime import datetime

# ! let's use default setting for efficient net on image net

base = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

data_path=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/
output=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/our-setting/

batchsize=64 #! use 256 for lower size
cd /data/duongdb/pytorch-image-models

# ! timm original setting on image net ... does not work for us. 
# python3 train.py $data_path --model efficientnet_b0 -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.3 --model-ema --model-ema-decay 0.9999 --remode pixel --reprob 0.2 --amp --lr .048 --filter_bias_and_bn --pretrained --num-classes 7 --topk 2 --output $output --scale 0.1 1.0 --eval-metric loss --amp

# ! loosely based on baseline in vienna 
python3 train.py --num-gpu 2 $data_path --model efficientnet_b7 -b $batchsize --sched cosine --epochs 850 --decay-epochs 50 --decay-rate 0.8 --opt nadam -j 8 --warmup-lr 1e-6 --weight-decay 0 --drop 0.3 --drop-connect 0.3 --model-ema --model-ema-decay 0.9999 --lr 0.0001 --filter_bias_and_bn --pretrained --num-classes 7 --topk 2 --output $output --scale 0.1 1.0 --eval-metric loss --amp --img-size 450 --remode pixel --reprob 0.1"""



os.chdir('/data/duongdb/HAM10000dataset/TrainDevTestRandState1/our-setting/')

case = {1: ' --aa ISIC2020 ', 
        2: ' --aa original --sampler ImbalancedDatasetSampler ',
        3: " --aa ISIC2020 --weighted_cross_entropy '30 19 9 87 8 1 70' --weighted_cross_entropy_eval ", 
        4: ' --aa ISIC2020 --sampler ImbalancedDatasetSampler '}

for k, val in case.items() : 
  if k not in [2]: # ! what to skip ?
    continue
  base2 = base + val 
  foutname = str(k)+'.sh' 
  fout = open(foutname, 'w')
  base2 = re.sub ('our-setting/','our-setting/'+str(k), base2)
  fout.write(base2 + "\n\n")
  fout.close() 
  #
  time.sleep(5)
  # os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:2 --mem=4g -c8 ' + foutname ) # k80


/data/duongdb/HAM10000dataset/TrainDevTestRandState1/efficientnet_b3/train/20200904-220137-efficientnet_b3-450