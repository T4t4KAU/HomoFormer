import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image deshadowing')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
import datetime
from utils import save_img
from losses import CharbonnierLoss
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from utils.image_utils import splitimage, mergeimage

from utils.loader import get_training_data, get_validation_data

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def _masked_psnr(pred, target, mask=None):
    if mask is None:
        return psnr_loss(target, pred, data_range=1.0)
    mask3 = np.repeat(mask, 3, axis=2).astype(bool)
    if mask3.sum() == 0:
        return np.nan
    mse = np.mean((pred[mask3] - target[mask3]) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def _masked_mae(pred, target, mask=None):
    diff = np.abs(pred - target) * 255.0
    if mask is None:
        return diff.mean()
    mask3 = np.repeat(mask, 3, axis=2).astype(bool)
    if mask3.sum() == 0:
        return np.nan
    return diff[mask3].mean()


def _mean_metric(values):
    values = [x for x in values if not np.isnan(x)]
    return sum(values) / len(values) if values else float('nan')


def validate_with_paper_metrics(model, val_loader, max_images, tile_size, tile_overlap):
    metrics = {
        'all_psnr': [], 'all_ssim': [], 'all_mae': [],
        'shadow_psnr': [], 'shadow_ssim': [], 'shadow_mae': [],
        'nonshadow_psnr': [], 'nonshadow_ssim': [], 'nonshadow_mae': [],
    }
    total = len(val_loader) if max_images == 0 else min(len(val_loader), max_images)
    for ii, data_val in enumerate(val_loader, 0):
        if max_images and ii >= max_images:
            break
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        mask = data_val[2].cuda()
        _, C, H, W = input_.shape

        tile = min(tile_size, H, W)
        if tile < 128:
            continue
        if tile >= 384:
            tile = 384
        elif tile >= 256:
            tile = 256
        else:
            tile = 128

        split_data, starts = splitimage(input_, crop_size=tile, overlap_size=tile_overlap)
        mask_data, _ = splitimage(mask, crop_size=tile, overlap_size=tile_overlap)
        restored_tiles = []
        for data, mask_ in zip(split_data, mask_data):
            with torch.cuda.amp.autocast():
                restored_tiles.append(model(data, mask_).cpu())
        restored = mergeimage(restored_tiles, starts, crop_size=tile, resolution=(1, C, H, W))

        rgb_restored = torch.clamp(restored, 0, 1).numpy().squeeze().transpose((1, 2, 0))
        rgb_gt = target.cpu().numpy().squeeze().transpose((1, 2, 0))
        bm = mask.cpu().numpy().squeeze()

        rgb_restored = cv2.resize(rgb_restored * 255.0, [256, 256], interpolation=cv2.INTER_AREA) / 255.0
        rgb_gt = cv2.resize(rgb_gt * 255.0, [256, 256], interpolation=cv2.INTER_AREA) / 255.0
        bm = cv2.resize(bm * 255.0, [256, 256], interpolation=cv2.INTER_AREA) / 255.0

        rgb_restored = np.clip(rgb_restored, 0, 1)
        rgb_gt = np.clip(rgb_gt, 0, 1)
        bm = np.where(bm < 0.001, np.zeros_like(bm), np.ones_like(bm))
        bm = np.expand_dims(bm, axis=2)
        non_bm = 1 - bm

        gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
        gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)

        metrics['all_psnr'].append(psnr_loss(rgb_gt, rgb_restored, data_range=1.0))
        metrics['all_ssim'].append(ssim_loss(gray_restored, gray_gt, data_range=1.0, channel_axis=None))
        metrics['all_mae'].append(_masked_mae(rgb_restored, rgb_gt))
        metrics['shadow_psnr'].append(_masked_psnr(rgb_restored, rgb_gt, bm))
        metrics['shadow_ssim'].append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), data_range=1.0, channel_axis=None))
        metrics['shadow_mae'].append(_masked_mae(rgb_restored, rgb_gt, bm))
        metrics['nonshadow_psnr'].append(_masked_psnr(rgb_restored, rgb_gt, non_bm))
        metrics['nonshadow_ssim'].append(ssim_loss(gray_restored * non_bm.squeeze(), gray_gt * non_bm.squeeze(), data_range=1.0, channel_axis=None))
        metrics['nonshadow_mae'].append(_masked_mae(rgb_restored, rgb_gt, non_bm))

        if (ii + 1) % 10 == 0 or (ii + 1) == total:
            print("Validated {}/{} images".format(ii + 1, total))

    return {k: _mean_metric(v) for k, v in metrics.items()}

######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()
print_network(model_restoration)

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1


# ######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()


######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=True, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = opt.eval_every
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
index = 0
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    for i, data in enumerate(train_loader, 0):
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        if epoch > 5:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_, mask)
            restored = torch.clamp(restored,0,1)
            loss = criterion(restored, target)
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()
        #### Evaluation ####
        if eval_now > 0 and (index+1)%eval_now==0 and i>0:
            with torch.no_grad():
                model_restoration.eval()
                val_metrics = validate_with_paper_metrics(
                    model_restoration, val_loader, opt.eval_max_images,
                    opt.eval_tile, opt.eval_tile_overlap)
                psnr_val_rgb = val_metrics['all_psnr']
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("[Ep %d it %d]" % (epoch, i))
                print("Shadow:     PSNR %.2f / SSIM %.3f / MAE %.2f" % (
                    val_metrics['shadow_psnr'], val_metrics['shadow_ssim'], val_metrics['shadow_mae']))
                print("Non-shadow: PSNR %.2f / SSIM %.3f / MAE %.2f" % (
                    val_metrics['nonshadow_psnr'], val_metrics['nonshadow_ssim'], val_metrics['nonshadow_mae']))
                print("All:        PSNR %.2f / SSIM %.3f / MAE %.2f" % (
                    val_metrics['all_psnr'], val_metrics['all_ssim'], val_metrics['all_mae']))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d]\n" % (epoch, i))
                    f.write("Shadow:     PSNR %.4f / SSIM %.4f / MAE %.4f\n" % (
                        val_metrics['shadow_psnr'], val_metrics['shadow_ssim'], val_metrics['shadow_mae']))
                    f.write("Non-shadow: PSNR %.4f / SSIM %.4f / MAE %.4f\n" % (
                        val_metrics['nonshadow_psnr'], val_metrics['nonshadow_ssim'], val_metrics['nonshadow_mae']))
                    f.write("All:        PSNR %.4f / SSIM %.4f / MAE %.4f ---- [best_Ep %d best_it %d Best_PSNR %.4f]\n" % (
                        val_metrics['all_psnr'], val_metrics['all_ssim'], val_metrics['all_mae'],
                        best_epoch, best_iter, best_psnr))
                model_restoration.train()
                torch.cuda.empty_cache()

        if index % 40 == 0:
            print("Epoch: {}\tIndex: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, index,
                                                                                          time.time() - epoch_start_time,
                                                                                          epoch_loss/40,
                                                                                          scheduler.get_lr()[0]))
            epoch_loss = 0
            epoch_start_time = time.time()
    scheduler.step()


    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
print("Now time is : ",datetime.datetime.now().isoformat())
