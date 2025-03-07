

# https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution#data-setup-assumes-the-kaggle-api-is-installed

cd /data/duongdb/ISIC2020-SkinCancerBinary
mkdir ./data-by-cdeotte
cd ./data-by-cdeotte
for input_size in 512 768 1024
do
  # kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
  # kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
  rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done

