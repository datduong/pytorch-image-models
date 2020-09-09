
import os,sys,re,pickle,time
from datetime import datetime

# ! valid 

# sinteractive --partition=gpu --gres=gpu:p100:1 --mem=4g -c4

base = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

data_path=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/test
base_path=/data/duongdb/HAM10000dataset/TrainDevTestRandState1/our-setting/
train_name=TRAIN_NAME

output=$base_path/$train_name/result_test.csv # path/name.csv
checkpoint=$base_path/$train_name/model_best.pth.tar # model_best.pth.tar averaged.pth 

batchsize=64
cd /data/duongdb/pytorch-image-models
python3 validate_no_label.py $data_path --model MODEL_NAME -b $batchsize -j 2 --config $base_path/$train_name/args.yaml --num-classes 7 --results-file $output --checkpoint $checkpoint --amp --use-ema --no-test-pool --has_eval_label

"""

os.chdir('/data/duongdb/HAM10000dataset/TrainDevTestRandState1/our-setting/')

case = {1: '20200909-005127-efficientnet_b1-450', 
        2: '20200909-005127-efficientnet_b2-450',
        3: '20200908-234403-efficientnet_b3-450', 
        4: ''}

for k, val in case.items() : 
	base2 = re.sub('TRAIN_NAME', '2/train/'+val, base) # train/
	base2 = re.sub('MODEL_NAME', val.split('-')[2], base2) # get name
	foutname = str(k)+'.'+val+'.sh'
	fout = open(foutname, 'w')
	fout.write(base2 + "\n\n")
	fout.close() 
	#
	# time.sleep(5)
	# os.system ( 'sbatch --partition=gpu --time=1-12:00:00 --gres=gpu:p100:1 --mem=6g -c8 ' + foutname ) # k80
	# os.system ( 'bash ' + foutname )
