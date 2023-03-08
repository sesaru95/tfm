import os
from monai.transforms import Resize

data_dir = os.getcwd()  # ruta directori de treball actual

n_folds = 5  # numero blocs en els que dividir el dataframe per entrenar/validar

image_sizes = [128, 128, 128]
R = Resize(image_sizes)
