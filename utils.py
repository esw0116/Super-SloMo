import cv2
import importlib
import math
import numpy as np
import torch
from typing import Any
from pytorch_msssim import ssim as ssim_pth


def meanshift(batch, rgb_mean, rgb_std, device, norm=True):
    if isinstance(rgb_mean, list):
        rgb_mean = torch.Tensor(rgb_mean)
    if isinstance(rgb_std, list):
        rgb_std = torch.Tensor(rgb_std)

    rgb_mean = rgb_mean.reshape(1,3,1,1).to(device)
    rgb_std = rgb_std.reshape(1,3,1,1).to(device)

    if norm:
        return (batch - rgb_mean) / rgb_std
    else:
        return (batch * rgb_std) + rgb_mean


def quantize(img, rgb_range=1):
    # change to 0~255
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
        img = img.mul(255).round().div(rgb_range)
    
    elif isinstance(img, np.ndarray):
        img = img.clip(0, 1)
        img = np.round(img*255)
    
    else:
        raise NotImplementedError()

    return img

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-255) arguments (masked inputs if mask is not None)
    '''
    diff = (pred - gt).div(255)

    if mask is not None:
        # consider only masked regions
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean()

    # mse can (surprisingly!) reach 0, which results in math domain error
    if mse <= 0:
        # print(mse)
        mse = 1e-5
        # breakpoint()
    return -10 * math.log10(mse)


def calc_ssim(img1, img2, datarange=255.):
    im1 = img1.numpy().transpose(1, 2, 0).astype(np.uint8)
    im2 = img2.numpy().transpose(1, 2, 0).astype(np.uint8)
    return compare_ssim(im1, im2, datarange=datarange, multichannel=True, gaussian_weights=True)


def calc_metrics(im_pred, im_gt, mask=None):
    q_im_pred = quantize(im_pred.data, rgb_range=1.)
    q_im_gt = quantize(im_gt.data, rgb_range=1.)
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())  # This function using SciPy compare_ssim() is very, very slow
    ssim = ssim_pth(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), val_range=255)
    return psnr, ssim


def eval_LPIPS(model, im_pred, im_gt):
    im_pred = 2.0 * im_pred - 1
    im_gt = 2.0 * im_gt - 1
    dist = model.forward(im_pred, im_gt)[0]
    return dist


def _eval_metrics(output, gt, psnrs, ssims, lpips=None, lpips_model=None, mask=None, psnrs_masked=None, ssims_masked=None):
    # PSNR should be calculated for each image
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs.update(psnr)
        ssims.update(ssim)
        if mask is not None:
            psnr_masked, ssim_masked = calc_metrics(output[b], gt[b], mask[b])
            psnrs_masked.update(psnr_masked)
            ssims_masked.update(ssim_masked)
        if lpips_model is not None:
            _lpips = eval_LPIPS(lpips_model, output[b].unsqueeze(0), gt[b].unsqueeze(0))
            lpips.update(_lpips)


def eval_metrics(output, gt, lpips_model=None):
    # PSNR should be calculated for each image
    psnrs = torch.zeros(gt.size(0), device=gt.device, dtype=gt.dtype)
    ssims = torch.zeros(gt.size(0), device=gt.device, dtype=gt.dtype)
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs[b] = psnr
        ssims[b] = ssim
        # if mask is not None:
        #     psnr_masked, ssim_masked = calc_metrics(output[b], gt[b], mask[b])
        #     psnrs_masked.update(psnr_masked)
        #     ssims_masked.update(ssim_masked)
        if lpips_model is not None:
            _lpips = eval_LPIPS(lpips_model, output[b].unsqueeze(0), gt[b].unsqueeze(0))
            # lpips.update(_lpips)

    return psnrs, ssims
