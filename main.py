from configuration.dataframes import df_seg
from configuration.datasets import SegDataset
from configuration.constants import transforms_show, n_folds
from run.segmentation import run as seg_run

import matplotlib.pyplot as plt

## Segmentació
# Visualització
seg_dataset = SegDataset(df_seg, 'show', transform=transforms_show)

n_img = 3
for i in range(n_img):
    fig, ax = plt.subplots(1, 3)
    for j in range(3):
        idx = i
        img, label = seg_dataset[idx]

        img = img.transpose(j + 1, 3)
        label = label.transpose(j + 1, 3)

        # agafem slice central
        img = img[:, :, :, 63]
        label = label[:, :, :, 63]

        # agrupem els 7 labels en 3 pq coincideixi el tamany de img i label
        label[0] = label[0] + label[3] + label[6]
        label[1] = label[1] + label[4]
        label[2] = label[2] + label[5]
        label = label[:3]

        img = img * 0.7 + label * 0.3  # per continuar en el rang [0, 1]
        img = img.transpose(0, 1).transpose(1, 2).squeeze()  # per tenir (128, 128, 3)
        ax[j].imshow(img)
        ax[j].set_axis_off()
        ax[j].set_title(seg_dataset.row['StudyInstanceUID'].split('.')[-1] + '_' + str(j+1))
    plt.show()

# Entrenament # ToDO Problemes treballant amb varios processos i assignacióo de memoria amb CUDA.
for i in range(n_folds):
    print('Running fold', i)
    seg_run(fold=i)


