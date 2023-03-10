from configuration.dataframes import df_seg
from configuration.datasets import SegDataset
from configuration.constants import transforms_train, transforms_valid

import matplotlib.pyplot as plt

# Mostra Dataset
seg_dataset = SegDataset(df_seg, 'train', transform=transforms_train)

a, b = 2, 4
fig, ax = plt.subplots(a, b)
for i in range(a):
    for p in range(b):
        idx = i * b + p
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

