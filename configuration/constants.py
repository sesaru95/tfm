import os
import monai.transforms as transforms

data_dir = os.getcwd()
log_dir = os.path.join(data_dir, 'logs')
model_dir = os.path.join(data_dir, 'models')
images_dir = os.path.join(data_dir, 'images')

n_folds = 5
roi_size = (160, 160, 160)
sw_batch_size = 4
batch_size = 4
num_workers = 4  # ToDO no accepta workers simultanis.
rate_learning = 3e-3
n_epochs = 1000

image_sizes = [128, 128, 128]
R = transforms.Resize(image_sizes)

## Transformacions
# transformacions de les imatges per tenir un dataset "més gran" per entrenar el model
transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    # inverteix l'ordre dels elements en un array
    transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image", "label"],
                           translate_range=[int(x * y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])],
                           padding_mode='zeros', prob=0.7),  # la imatge s'inclina en el pla de la imatge
    transforms.RandGridDistortiond(keys=("image", "label"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
    # distorsiona la imatge
])

# per validar/mostrar no necesitem ampliar el dataset amb aquestes transformacions
transforms_valid = transforms.Compose([
])
transforms_show = transforms.Compose([
])

# tranformacions a format one hot per calcular la metrica
post_pred = transforms.Compose([
    transforms.AsDiscrete(argmax=True, to_onehot=7)
])

post_label = transforms.Compose([
    transforms.AsDiscrete(to_onehot=7)
])

# imatges detectades amb els labels invertits en l'eix Z
revert_list = [
    '1.2.826.0.1.3680043.1363',
    '1.2.826.0.1.3680043.20120',
    '1.2.826.0.1.3680043.2243',
    '1.2.826.0.1.3680043.24606',
    '1.2.826.0.1.3680043.32071'
]
