from configuration.dataframes import df_seg
from configuration.datasets import SegDataset
from configuration.constants import transforms_train, transforms_valid

import matplotlib.pyplot as plt

seg_dataset = SegDataset(df_seg, 'train', transform=transforms_valid)

fig, ax = plt.subplots(2, 3)
for i in range(2):
    for p in range(3):
        idx = i * 3 + p  # la 4a imatge peta per problemes amb la memoria
        img, label = seg_dataset[idx]

        # agafem slice central
        img = img[:, :, :, 63]
        label = label[:, :, :, 63]

        # agrupem els 7 labels en 3 pq coincideixi el tamany de img i label
        label[0] = label[0] + label[3] + label[6]
        label[1] = label[1] + label[4]
        label[2] = label[2] + label[5]
        label = label[:3]

        img = img * 0.7 + label * 0.3  # per continuar en el rang [0, 1]
        img = img.transpose(0, 1).transpose(1, 2).squeeze()  # per tenir (128, 128, 3, 1)
        ax[i, p].imshow(img)
        ax[i, p].set_title(seg_dataset.row['StudyInstanceUID'].split('.')[-1])

plt.show()

