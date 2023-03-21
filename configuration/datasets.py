from configuration.constants import image_sizes, R, revert_list
import os
import cv2
import pydicom
import numpy as np
from glob import glob
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_dicom_slice(path):  # funció per llegir una imatge DICOM donada una ruta i adaptar-ne el tamany
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (image_sizes[0], image_sizes[1]), interpolation=cv2.INTER_LINEAR)
    return data


def load_dicom_scan(
        path):  # funció per llegir 128 imatges de la ruta (a intervals regulars), les concatena i normalitza
    l_paths = sorted(glob(os.path.join(path, "*")),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))

    n_paths = len(l_paths)
    indices = np.quantile(list(range(n_paths)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    l_paths = [l_paths[i] for i in indices]

    images = []
    for filename in l_paths:
        images.append(load_dicom_slice(filename))
    images = np.stack(images, -1)  # ajuntem les imatges per l'eix z

    # [0., 1.]
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)  # així evitem divisions per 0
    # [0, 255]
    images = (images * 255).astype(np.uint8)

    return images


# funció per obtenir la imatge 3D amb canal de color i 7 imatges amb un label cada una
def load_sample(row, exists_seg_file=True):
    image = load_dicom_scan(row.image_folder)
    dim = image.ndim
    # afegeix una quarta dimensió al array de la imatge triplicant-la (128, 128, 128) -> (3, 128, 128, 128)
    # aquesta dimensió correspon als canals de color
    if dim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0)

    # 7 imatges, cada una informa un label diferent i la resta a zero
    if exists_seg_file:
        seg_data = nib.load(row.file_dir).get_fdata()
        shape = seg_data.shape
        seg_data = seg_data.transpose(1, 0, 2)[::-1, :, ::-1]  # correcció orientació (eix X reflectit i eix Z girat)
        label = np.zeros((7, shape[0], shape[1], shape[2]))

        for c_id in range(7):
            label[c_id] = (seg_data == (c_id + 1))
        label = label.astype(np.uint8) * 255
        label = R(label).numpy()  # (7, 128, 128, 128)

        return image, label
    else:
        return image


class SegDataset(Dataset):
    def __init__(self, df, mode, transform):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        self.row = self.df.iloc[index]

        image, label = load_sample(self.row, exists_seg_file=True)

        if self.row.StudyInstanceUID in revert_list:
            label = label[:, :, :, ::-1]

        res = self.transform({'image': image, 'label': label})  # transformacions per 'ampliar' dataset
        # [0..255] -> [0..1] format adequat per la funció imshow
        image = res['image'] / 255.
        label = res['label']
        label = (label > 127).astype(np.float32)

        image, label = torch.tensor(image).float(), torch.tensor(label).float()

        return image, label
