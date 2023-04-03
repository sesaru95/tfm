import numpy as np
import torch
from tqdm import tqdm
from monai.data import decollate_batch
from model.segmentation import loss_function, dice_metric, device
from monai.inferers import sliding_window_inference

from configuration.constants import post_pred, post_label, roi_size, sw_batch_size


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            # loss
            outputs = model(images)
            loss = loss_function(outputs, labels)
            valid_loss.append(loss.item())
            bar.set_description(f'valid_loss:{np.mean(valid_loss):.4f}')

            # metric
            outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            dice_metric(y_pred=outputs, y=labels)

        # afegeix la mitja final resultant
        metric = dice_metric.aggregate().item()
        print('best dc:', np.max(metric))

        # reinicialitza la funció
        dice_metric.reset()

    return np.mean(valid_loss), metric
