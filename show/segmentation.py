from configuration.datasets import SegDataset
from configuration.constants import images_dir

import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def show(df, transform):
    os.makedirs(images_dir, exist_ok=True)

    seg_dataset = SegDataset(df=df, mode='show', transform=transform)

    n_img = len(seg_dataset)
    bar = tqdm(range(n_img))
    for i in bar:
        fig, ax = plt.subplots(1, 3)

        for j in range(3):
            img, label = seg_dataset[i]
            img = img.transpose(j + 1, 3)
            label = label.transpose(j + 1, 3)

            # agafem slice central
            img = img[:, :, :, 63]
            label = label[:, :, :, 63]

            # agrupem els 7 labels en 3 pq coincideixi el tamany de img i label
            label[0] = label[0] + label[3] + label[5] + label[6]
            label[1] = label[1] + label[3] + label[4] + label[6]
            label[2] = label[2] + label[4] + label[5] + label[6]
            label = label[:3]

            img = img * 0.7 + label * 0.3  # per continuar en el rang [0, 1]
            img = img.transpose(0, 1).transpose(1, 2).squeeze()  # per tenir (128, 128, 3)

            fig_tittle = seg_dataset.row.StudyInstanceUID
            image_file = os.path.join(images_dir, f'{fig_tittle}.png')

            ax[j].imshow(img)
            ax[j].set_axis_off()
            ax[j].set_title(fig_tittle.split('.')[-1] + '_' + str(j + 1))

        plt.savefig(image_file)
