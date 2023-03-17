import os
import gc
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from configuration.dataframes import df_seg
from configuration.datasets import SegDataset
from configuration.constants import transforms_train, transforms_valid, batch_size, num_workers, rate_learning, \
    n_epochs, log_dir, model_dir
from model.segmentation import model
from train import train_func
from valid import valid_func


def run(fold):
    # Files
    log_file = os.path.join(log_dir, f'logs_fold{fold}.txt')
    model_file = os.path.join(model_dir, f'fold{fold}_best_metric_model.pth')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # DataFrames
    df_train = df_seg[df_seg['fold'] != fold].reset_index(drop=True)
    df_valid = df_seg[df_seg['fold'] == fold].reset_index(drop=True)

    # DataSets
    ds_train = SegDataset(df_train, 'train', transform=transforms_train)
    ds_valid = SegDataset(df_valid, 'valid', transform=transforms_valid)
    print(f'dataset train size: {len(ds_train)}, dataset valid size: {len(ds_train)}')

    # DataLoader
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    optimizer = optim.AdamW(model.parameters(), lr=rate_learning)
    scaler = torch.cuda.amp.GradScaler()
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_epochs)

    # Train
    metric_best = 0.
    epoch_metric_best = 0
    loss_values = []
    metric_values = []
    for epoch in range(1, n_epochs + 1):
        scheduler_cosine.step(epoch - 1)  # actualitza learning rate

        print(time.ctime(), 'Starting epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, metric = valid_func(model, loader_valid)

        loss_values.append(train_loss)
        metric_values.append(metric)

        if metric > metric_best:
            print(f'Saving new best metric model ...')
            torch.save(model.state_dict(), model_file)
            print("Saved new best metric model")
            metric_best = metric
            epoch_metric_best = epoch + 1

        content = time.ctime() + ' ' + f'fold {fold}, current epoch {epoch}, rate learning: {optimizer.param_groups[0]["lr"]:.7f}, ' \
                                       f'train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {metric :.6f}, ' \
                                       f'best mean dice: {metric_best:.6f}  at epoch: {epoch_metric_best}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        print(time.ctime(), 'Finishing epoch:', epoch)

    # Plot loss and metric
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(loss_values))]
    y = loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()

    # neteja
    del model
    torch.cuda.empty_cache()
    gc.collect()
