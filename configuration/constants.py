import os
import monai.transforms as transforms

data_dir = os.getcwd()  # ruta directori de treball actual

n_folds = 5  # numero blocs en els que dividir el dataframe per entrenar/validar

image_sizes = [128, 128, 128]
R = transforms.Resize(image_sizes)

# transformacions de les imatges per tenir un dataset "més gran" per entrenar el model
transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1), # inverteix l'ordre dels elements en un array
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image", "label"], translate_range=[int(x*y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.7),  # la imatge s'inclina en el pla de la imatge
    transforms.RandGridDistortiond(keys=("image", "label"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"), # distorsiona la imatge
])

# per validar no necesitem ampliar el dataset amb aquestes transformacions
transforms_valid = transforms.Compose([
])
