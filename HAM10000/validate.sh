


# data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/ # train/valid
# data_path=/data/duongdb/HAM10000dataset/ISIC2018_Task3_Test_Input/transform/ # test
data_path=/data/duongdb/HAM10000dataset/ISIC2018_Task3_Test_Input/AugmentTransform/ # test
batchsize=64
cd /data/duongdb/pytorch-image-models
base_path=/data/duongdb/HAM10000dataset/efficientnet_b2/train/
train_name='20200820-003052-efficientnet_b2-260'
output=$base_path/$train_name/result_test.csv # path/name.csv
checkpoint=$base_path/$train_name/model_best.pth.tar

python3 validate_no_label.py $data_path --model efficientnet_b2 -b $batchsize -j 4 --num-classes 7 --results-file $output --checkpoint $checkpoint --amp --use-ema --average_augment




# ! do test 
# data_path=/data/duongdb/HAM10000dataset/ImagesLabelFolderRandomSplit/ # train/valid
data_path=/data/duongdb/HAM10000dataset/ISIC2018_Task3_Test_Input/original/ # test
# data_path=/data/duongdb/HAM10000dataset/ISIC2018_Task3_Test_Input/transform/ # test
# data_path=/data/duongdb/HAM10000dataset/ISIC2018_Task3_Test_Input/AugmentTransform/ # test
batchsize=64
cd /data/duongdb/pytorch-image-models
base_path=/data/duongdb/HAM10000dataset/efficientnet_b0/train/
train_name='20200825-193738-efficientnet_b0-224'

# ! average check point 
# python3 avg_checkpoints.py --input $base_path/$train_name --output $base_path/$train_name/averaged.pth


output=$base_path/$train_name/result_test.csv # path/name.csv
checkpoint=$base_path/$train_name/averaged.pth # model_best.pth.tar

python3 validate_no_label.py $data_path --model efficientnet_b0 -b $batchsize -j 2 --num-classes 7 --results-file $output --checkpoint $checkpoint --amp --use-ema --create_classifier_layerfc

# average_augment

cd $base_path/$train_name
