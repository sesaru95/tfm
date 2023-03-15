import os
import pandas as pd
from sklearn.model_selection import KFold
from configuration.constants import data_dir, n_folds

# Segmentation
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # df amb el contingut de trains.csv

files_name = os.listdir(f'{data_dir}/segmentations')  # llista amb el nom dels fitxers a segmentations

df_files = pd.DataFrame({
    'file_name': files_name,
})  # df amb el nom dels fitxers de segmentations

df_files['StudyInstanceUID'] = df_files['file_name'].apply(lambda x: x[:-4])  # afegim l'UID de cada fitxer al df
df_files['file_dir'] = df_files['file_name'].apply(
    lambda x: os.path.join(data_dir, 'segmentations', x))  # afegim la ruta de cada fitxer al df

df = df_train.merge(df_files, on='StudyInstanceUID', how='left')  # merge dels dos df

df['image_folder'] = df['StudyInstanceUID'].apply(lambda x: os.path.join(data_dir, 'train_images', x))  # afegim al
# df la ruta de les carpetes de train_images
df['file_dir'].fillna('', inplace=True)  # substituim els NaN per espais en blanc

df_seg = df.query('file_dir != ""').reset_index(drop=True)  # eliminem files amb file_dir buit

# dividim el df en n_fold blocs on un es el conjunt de validacio i els altres son els d'entrenament
kf = KFold(n_folds)
df_seg['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df_seg)):
    df_seg.loc[valid_idx, 'fold'] = fold

__all__ = [
    "df_seg"
]
