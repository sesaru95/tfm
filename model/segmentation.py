import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric

# fem servir un model 3D del packet MONAI: UNet
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = nn.BCEWithLogitsLoss()
dice_metric = DiceMetric(include_background=False, reduction="mean")


