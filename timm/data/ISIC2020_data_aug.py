import albumentations

def ISIC2020_get_transforms(image_size, is_training):
    # https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py

    if is_training : 
        return  albumentations.Compose([
                albumentations.Transpose(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightness(limit=0.2, p=0.75),
                albumentations.RandomContrast(limit=0.2, p=0.75),
                albumentations.OneOf([
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                albumentations.OneOf([
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                    albumentations.ElasticTransform(alpha=3),
                ], p=0.7),

                albumentations.CLAHE(clip_limit=4.0, p=0.7),
                albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                albumentations.Resize(image_size, image_size),
                albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
                albumentations.Normalize()
                ])

    else: 
        return  albumentations.Compose([
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize()
                ])


