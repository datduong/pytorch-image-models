
import os,sys,re,pickle,time

base = """#!/bin/bash
data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderOriginalFormatTrainDevRandomState1/
output=/data/duongdb/HAM10000dataset/MODEL-NAME/
batchsize=64
cd /data/duongdb/pytorch-image-models

python3 train.py $data_path --model MODEL-NAME -b $batchsize --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 4 --warmup-lr 1e-6 --weight-decay 0 --last_layer_weight_decay 0.0001 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa original --vflip 0.5 --remode pixel --reprob 0 --amp --lr LEARNING-RATE --classification_layer_name 'classifier' --filter_bias_and_bn --pretrained --num-classes 7 --output $output --weighted_cross_entropy '30.62691 19.48443 9.11282 87.08695 8.99820 1.49366 70.52816' --create_classifier_layerfc
"""

# ! their policy rand-m9-mstd0.5

os.chdir('/data/duongdb/pytorch-image-models/HAM10000')

base_count = 1

script_base_array = [ 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2' ]
LR = [0.001, 0.0001]

for script_base in script_base_array: 
  for index, val in enumerate(LR) :
    index = index + base_count
    base2 = re.sub( 'LEARNING-RATE' , str(val), base )
    base2 = re.sub( 'MODEL-NAME' , script_base, base2 )
    fout = open(script_base+'.'+str(index)+'.sh', 'w')
    fout.write(base2)
    fout.close() 
    #
    time.sleep(1)
    os.system ( 'sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:k80:1 --mem=8g -c4 ' + script_base+'.'+str(index)+'.sh' )
    
