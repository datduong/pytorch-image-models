


# ! do test 

sinteractive --partition=gpu --gres=gpu:k80:1 --mem=4g -c4
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate base

# module load python/3.7

data_path=/data/duongdb/ISIC2019-SkinCancer8Labels/TrainDevTestRandState1/test

batchsize=32
cd /data/duongdb/pytorch-image-models
base_path=/data/duongdb/ISIC2019-SkinCancer8Labels/TrainDevTestRandState1/efficientnet_b0/train

# ! average check point 
# python3 avg_checkpoints.py --input $base_path/$train_name --output $base_path/$train_name/averaged.pth

train_name='20200901-185604-efficientnet_b0-224'
output=$base_path/$train_name/result_test.csv # path/name.csv
checkpoint=$base_path/$train_name/model_best.pth.tar # model_best.pth.tar averaged.pth 

python3 validate_no_label.py $data_path --model efficientnet_b0 -b $batchsize -j 4 --config $base_path/$train_name/args.yaml --num-classes 9 --results-file $output --checkpoint $checkpoint --amp --use-ema --no-test-pool --has_eval_label

# --has_eval_label
# --average_augment
# --aa original
# --no-test-pool

cd $base_path/$train_name



