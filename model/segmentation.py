import torch
import torch.nn as nn
from monai.networks.nets import UNet, densenet121
from monai.networks.layers import Norm
from monai.metrics import DiceMetric

# fem servir un model 3D del packet MONAI: UNet/DenseNet
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
monai_desnet_model = densenet121(
    spatial_dims=3,
    in_channels=3,
    out_channels=8,
).to(device)  # ToDo aquest model falla per input diferent target.
monai_unet_model = UNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=7,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = nn.BCEWithLogitsLoss()
dice_metric = DiceMetric(include_background=False, reduction="mean")


