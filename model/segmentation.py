import torch
import torch.nn as nn
from monai.networks.nets import densenet121
from monai.metrics import DiceMetric

# fem servir un model 3D del packet MONAI: UNet
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
model = densenet121(
    spatial_dims=3,
    in_channels=3,
    out_channels=8,
).to(device)
loss_function = nn.BCEWithLogitsLoss()
dice_metric = DiceMetric(include_background=False, reduction="mean")


