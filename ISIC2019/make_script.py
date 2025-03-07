
import os,sys,re,pickle,time
from datetime import datetime

base = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate base

# module load python/3.7

data_path=/data/duongdb/ISIC2019-SkinCancer8Labels/TrainDevTestRandState1/
output=/data/duongdb/ISIC2019-SkinCancer8Labels/TrainDevTestRandState1/MODEL-NAME/

batchsize=64
cd /data/duongdb/pytorch-image-models


# python3 train.py $data_path --model MODEL-NAME -b $batchsize --sched step --epochs 450 --decay-rate 0.5 --opt nadam -j 8 --warmup-lr 1e-6 --weight-decay 0 --last_layer_weight_decay 0.0001 --drop DROPOUT --drop-connect DROPOUT --model-ema --model-ema-decay 0.9999 --aa ISIC2020 --vflip 0.5 --remode pixel --reprob 0 --lr LEARNING-RATE --classification_layer_name 'classifier' --filter_bias_and_bn --pretrained --num-classes 9 --topk 2 --output $output --create_classifier_layerfc --scale 0.1 1.0 --eval-metric loss --sampler ImbalancedDatasetSampler

python3 train.py $data_path --model MODEL-NAME -b $batchsize --sched cosine --epochs 450 --decay-epochs 50 --decay-rate 0.8 --opt nadam -j 8 --warmup-lr 1e-6 --weight-decay 0 --drop DROPOUT --drop-connect DROPOUT --model-ema --model-ema-decay 0.9999 --lr LEARNING-RATE --filter_bias_and_bn --pretrained --num-classes 9 --topk 2 --output $output --scale 0.1 1.0 --eval-metric loss --amp --weighted_cross_entropy '29 8 10 106 6 2 40 1 100' --weighted_cross_entropy_eval --aa ISIC2020 --remode pixel --reprob 0.1

"""

# --weighted_cross_entropy '29 8 10 106 6 2 40 1 100' 
# --amp

os.chdir('/data/duongdb/pytorch-image-models/ISIC2019')


script_base_array = [ 'efficientnet_b0' ] #  'efficientnet_b0' , 'efficientnet_b1', 'efficientnet_b2' 'inception_v3' nadam
LR = [0.00001]

for script_base in script_base_array: 
  for dropout in [0.3] : # [0.1,0.3]: 
    for index, val in enumerate(LR) :
      index = int(datetime.now().strftime('%Y%m%d%H%M%S'))
      base2 = re.sub( 'LEARNING-RATE' , str(val), base )
      base2 = re.sub( 'MODEL-NAME' , script_base, base2 )
      base2 = re.sub( 'DROPOUT' , str(dropout), base2 )
      foutname = script_base+'.'+str(index)+'.sh' # str(val)+'.'+str(dropout)+'.'+
      fout = open(foutname, 'w')
      fout.write(base2)
      fout.close() 
      #
      time.sleep(3)
      if 'ISIC2020' in base2: 
        os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=8g -c8 ' + foutname ) # k80
      else: 
        os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=4g -c8 ' + foutname ) # k80

     
#

