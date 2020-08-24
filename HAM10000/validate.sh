


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



