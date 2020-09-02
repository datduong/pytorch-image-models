
import os,sys,re,pickle,time
from datetime import datetime

base = """#!/bin/bash

module load python/3.7

# data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderOriginalFormatTrainDevRandomState1/
data_path=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/
output=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/MODEL-NAME/

batchsize=64
cd /data/duongdb/pytorch-image-models

python3 train.py $data_path --model MODEL-NAME -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 0 --last_layer_weight_decay 0.0001 --drop DROPOUT --drop-connect DROPOUT --model-ema --model-ema-decay 0.9999 --aa original --vflip 0.5 --remode pixel --reprob 0 --amp --lr LEARNING-RATE --classification_layer_name 'classifier' --filter_bias_and_bn --pretrained --num-classes 7 --topk 2 --output $output --weighted_cross_entropy '30 19 9 87 8 1 70' --create_classifier_layerfc --scale 0.1 1.0 --eval-metric loss 

"""

# --weighted_cross_entropy '30.62691 19.48443 9.11282 87.08695 8.99820 1.49366 70.52816' 
# --decay-rate 0.5 --opt nadam -j 6 --warmup-lr 1e-6 --weight-decay 0
# --img-size 450
# --decay-epochs 2.4
# where_resume=/data/duongdb/HAM10000dataset/efficientnet_b2/train/20200829-170827-efficientnet_b2-450/
# --resume $where_resume/last.pth.tar
# --drop 0.2 --drop-connect 0.2
# rmsproptf nadam
# their policy rand-m9-mstd0.5
# --aug_eval_data

os.chdir('/data/duongdb/pytorch-image-models/HAM10000')


script_base_array = [ 'efficientnet_b0' ] #  'efficientnet_b0' , 'efficientnet_b1', 'efficientnet_b2' 'inception_v3' nadam
LR = [0.001]

for script_base in script_base_array: 
  for dropout in [0.2] : # [0.1,0.3]: 
    for index, val in enumerate(LR) :
      index = index + int(datetime.now().strftime('%Y%m%d%H%M%S'))
      base2 = re.sub( 'LEARNING-RATE' , str(val), base )
      base2 = re.sub( 'MODEL-NAME' , script_base, base2 )
      base2 = re.sub( 'DROPOUT' , str(dropout), base2 )
      foutname = str(val)+'.'+str(dropout)+'.'+script_base+'.'+str(index)+'.sh'
      fout = open(foutname, 'w')
      fout.write(base2)
      fout.close() 
      #
      time.sleep(3)
      os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=5g -c6 ' + foutname ) # k80

     
#

