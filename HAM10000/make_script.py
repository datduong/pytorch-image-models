
import os,sys,re,pickle,time
from datetime import datetime

base = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

# module load python/3.7

# data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderOriginalFormatTrainDevRandomState1/
data_path=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/
output=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/MODEL-NAME/

batchsize=64
cd /data/duongdb/pytorch-image-models

python3 train.py $data_path --model MODEL-NAME -b $batchsize --sched cosine --epochs 450 --decay-epochs 50 --decay-rate 0.8 --opt nadam -j 8 --warmup-lr 1e-6 --weight-decay 0 --drop DROPOUT --drop-connect DROPOUT --model-ema --model-ema-decay 0.9999 --lr LEARNING-RATE --filter_bias_and_bn --pretrained --num-classes 7 --topk 2 --output $output --scale 0.1 1.0 --eval-metric loss --amp --weighted_cross_entropy --weighted_cross_entropy_eval '30 19 9 87 8 1 70' --aa ISIC2020 --remode pixel --reprob 0.1

"""

# --sched cosine --epochs 200 --lr 0.05
# --sampler ImbalancedDatasetSampler 
# --sched step --last_layer_weight_decay 0.0001 --create_classifier_layerfc --classification_layer_name 'classifier'
# --amp
# --weighted_cross_entropy '30 19 9 87 8 1 70'
# --aa original
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
LR = [0.0005]

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
        os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=6g -c8 ' + foutname ) # k80
      else: 
        os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=6g -c8 ' + foutname ) # k80

     
#

# sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:p100:2 --mem=4g -c8 efficientnet_b3.20200904154748.sh
--resume /data/duongdb/HAM10000dataset/TrainDevTestRandState1/efficientnet_b3/train/20200904-220137-efficientnet_b3-450/last.pth.tar

sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=4g -c8 efficientnet_b0.20200904205005.sh
