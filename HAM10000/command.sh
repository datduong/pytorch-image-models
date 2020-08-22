
cd /data/duongdb/pytorch-image-models/HAM10000

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:k80:2 --mem=8g -c8 train_basic.sh

sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:1 --mem=8g -c4 EfficientnetB2wt.sh

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:k80:1 --mem=8g -c8 EfficientnetB0.sh

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:k80:1 --mem=8g -c8 EfficientnetB0wt.sh

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:k80:1 --mem=8g -c8 EfficientnetB1.sh

sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:k80:1 --mem=8g -c4 EfficientnetB1wt.sh

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:p100:1 --mem=8g -c8 InceptionV3.sh

sbatch --partition=gpu --time=3-12:00:00 --gres=gpu:p100:2 --mem=8g -c8 InceptionV3wt.sh


# -------------------------------- interactive ------------------------------- #

sinteractive --partition=gpu --gres=gpu:k80:1 --mem=6g -c6
