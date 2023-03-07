from configuration.constants import image_sizes
import os
import cv2
import pydicom
import numpy as np
from glob import glob


def load_dicom_slice(path):  # funció per llegir una imatge DICOM donada una ruta i adaptar-ne el tamany
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (image_sizes[0], image_sizes[1]), interpolation=cv2.INTER_LINEAR)
    return data


def load_dicom_scan(path):  # llegeix 128 imatges de la ruta (a intervals regulars), les concatena i normalitza
    l_paths = sorted(glob(os.path.join(path, "*")),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))

    n_paths = len(l_paths)
    indices = np.quantile(list(range(n_paths)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    l_paths = [l_paths[i] for i in indices]

    images = []
    for filename in l_paths:
        images.append(load_dicom_slice(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images
