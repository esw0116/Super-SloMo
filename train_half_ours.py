
#[Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import superslomo_half as superslomo
from model.extraction import center, ends
from data import gopro_blur_half as gopro_blur
import dataloader
from utils import quantize, eval_metrics, meanshift

from copy import deepcopy
from math import log10
import datetime
import os, sys
import numpy as np
import pandas as pd
import tqdm


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", action='store_true', help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--test_only", action='store_true', help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--add_blur", action='store_true', help='Add blurry image')
parser.add_argument("--extract11", action='store_true', help='Add blurry image')
parser.add_argument("--epochs", type=int, default=40, help='number of epochs to train. Default: 200.')
parser.add_argument("--seq_len", type=int, default=11, help='number of frames that composes a sequence.')
parser.add_argument("--train_batch_size", type=int, default=8, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=4, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[20, 30], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=1000, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=1, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
parser.add_argument("--amp", action='store_true', help='If True, use mixed precision.')
args = parser.parse_args()

### For visualizing loss and interpolated frames

writer = SummaryWriter('log')
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

scaler = torch.cuda.amp.GradScaler()

###Initialize flow computation and arbitrary-time flow interpolation CNNs.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = superslomo.UNet(6, 4)
flowComp.to(device)
if args.add_blur:
    ArbTimeFlowIntrp = superslomo.UNet(23, 5)
else:
    ArbTimeFlowIntrp = superslomo.UNet(20, 5)
ArbTimeFlowIntrp.to(device)


###Initialze backward warpers for train and validation datasets

trainFlowBackWarp      = superslomo.backWarp(352, 352, device)
trainFlowBackWarp      = trainFlowBackWarp.to(device)
validationFlowBackWarp = superslomo.backWarp(1280, 704, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)


### Load Pretrained Extraction Models

center_estimation = center.Center()
border_estimation = ends.Ends()

if not args.extract11:
    print('Estimation Network: best_gopro07.ckpt')
    pretrained_weight = torch.load('pretrained_models/best_gopro07.ckpt')['state_dict']
else:
    print('Estimation Network: best_gopro11.ckpt')
    pretrained_weight = torch.load('pretrained_models/best_gopro11.ckpt')['state_dict']

center_state_dict = {}
ends_state_dict = {}
for key, value in pretrained_weight.items():
    if key.startswith('center_est.'):
        center_state_dict[key[11:]] = value
    elif key.startswith('gen.'):
        ends_state_dict[key[4:]] = value

# print(center_state_dict.keys())

center_estimation.load_state_dict(center_state_dict)
border_estimation.load_state_dict(ends_state_dict)
center_estimation = center_estimation.to(device)
border_estimation = border_estimation.to(device)

### Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

# GoPro
trainset = gopro_blur.GoPro(root=args.dataset_root, transform=transform, seq_len=args.seq_len, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

validationset = gopro_blur.GoPro(root=args.dataset_root, transform=transform, randomCropSize=(1280, 704), seq_len=args.seq_len, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

print(trainset, validationset)


###Create transform to display image from tensor

negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


###Utils

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


###Loss and Optimizer

seq_len = args.seq_len
ctr_idx = seq_len // 2
L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()
compare_ftn = nn.L1Loss(reduction='none')

params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

optimizer = optim.Adam(params, lr=args.init_learning_rate)

# scheduler to decrease learning rate by a factor of 10 at milestones.
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=3, verbose=True)

###Initializing VGG16 model for perceptual loss

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False


### Validation function

def validate():
    # For details see training.
    psnr = 0
    tloss = 0
    flag = 1
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex, _) in enumerate(validationloader, 0):
            # frame0, frameT, frame1 = validationData

            # I0 = frame0.to(device)
            # I1 = frame1.to(device)
            # IFrame = frameT.to(device)

            blurred_img = torch.zeros_like(validationData[0])
            for image in validationData:
                blurred_img += image
            blurred_img /= len(validationData)
            blurred_img = blurred_img.to(device)
            
            blurred_img = meanshift(blurred_img, mean, std, device, False)
            center = center_estimation(blurred_img)
            start, end = border_estimation(blurred_img, center)
            start = meanshift(start, mean, std, device, True)
            end = meanshift(end, mean, std, device, True)
            center = meanshift(center, mean, std, device, True)
            blurred_img = meanshift(blurred_img, mean, std, device, True)

            frame0 = validationData[0].to(device)
            frame1 = validationData[-1].to(device)

            batch_size = blurred_img.shape[0]
            parallel = torch.mean(compare_ftn(start, frame0) + compare_ftn(end, frame1), dim=(1,2,3))
            cross = torch.mean(compare_ftn(start, frame1) + compare_ftn(end, frame0), dim=(1,2,3))

            # I0 = torch.zeros_like(blurred_img)
            IFrame = torch.zeros_like(blurred_img)
            choose_start = torch.zeros(batch_size).to(device)

            for b in range(batch_size):
                if (validationFrameIndex[b] < (ctr_idx - 1) and parallel[b] <= cross[b]) or (validationFrameIndex[b] > (ctr_idx - 1) and parallel[b] > cross[b]):
                    # I0[b] = start[b]
                    choose_start[b] = 1
                else:
                    # I0[b] = end[b]
                    choose_start[b] = 0

                IFrame[b] = validationData[validationFrameIndex[b]+1][b]
                if validationFrameIndex[b] > (ctr_idx - 1):
                    validationFrameIndex[b] = seq_len - 3 - validationFrameIndex[b]

            choose_start = choose_start.reshape((-1, 1, 1, 1))
            I0 = choose_start * start + (1-choose_start) * end
            I1 = center

            if args.amp:
                with torch.cuda.amp.autocast():
                    flowOut = flowComp(torch.cat((I0, I1), dim=1))
                    F_0_1 = flowOut[:,:2,:,:]
                    F_1_0 = flowOut[:,2:,:,:]

                    fCoeff = superslomo.getFlowCoeff(validationFrameIndex, device, seq_len)

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
                    
                    if args.add_blur:
                        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
                    else:
                        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0
                    
                    g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
                    
                    wCoeff = superslomo.getWarpCoeff(validationFrameIndex, device, seq_len)
                    
                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                    
                    #loss
                    recnLoss = L1_lossFn(Ft_p, IFrame)
                    
                    prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
                    
                    warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
                
                    loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
                    loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
                    loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
                    
                    loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            else:
                flowOut = flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]

                fCoeff = superslomo.getFlowCoeff(validationFrameIndex, device, seq_len)

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
                
                if args.add_blur:
                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
                else:
                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                
                g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
                
                wCoeff = superslomo.getWarpCoeff(validationFrameIndex, device, seq_len)
                
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                #loss
                recnLoss = L1_lossFn(Ft_p, IFrame)
                
                prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
                
                warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
            
                loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
                loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
                loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
                
                loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            tloss += loss.item()
            
            # For tensorboard
            if flag:
                retImg = torchvision.utils.make_grid([revNormalize(frame0.cpu()[0]), revNormalize(IFrame.cpu()[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1.cpu()[0])], padding=10)
                flag = 0
            
            #psnr
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))

            # Make benchmark csv file

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg


def test():
    df_column = ['Name']
    df_column.extend([str(i) for i in range(1, seq_len + 1)])

    df = pd.DataFrame(columns=df_column)

    psnr_array = np.zeros((0, seq_len))
    ssim_array = np.zeros((0, seq_len))

    tqdm_loader = tqdm.tqdm(validationloader, ncols=80)

    imgsave_folder = os.path.join(args.checkpoint_dir, 'Saved_imgs')
    if not os.path.exists(imgsave_folder):
        os.mkdir(imgsave_folder)

    with torch.no_grad():
        for validationIdx, (validationData, _, validationFile) in enumerate(tqdm_loader):
            
            blurred_img = torch.zeros_like(validationData[0])
            for image in validationData:
                blurred_img += image
            blurred_img /= len(validationData)
            blurred_img = blurred_img.to(device)
            batch_size = blurred_img.shape[0]

            blurred_img = meanshift(blurred_img, mean, std, device, False)
            center = center_estimation(blurred_img)
            start, end = border_estimation(blurred_img, center)

            # start, end, center = quantize(start, rgb_range=255), quantize(end, rgb_range=255), quantize(center, rgb_range=255)

            start = meanshift(start, mean, std, device, True)
            end = meanshift(end, mean, std, device, True)
            center = meanshift(center, mean, std, device, True)
            blurred_img = meanshift(blurred_img, mean, std, device, True)

            frame0 = validationData[0].to(device)
            frame1 = validationData[-1].to(device)
            
            batch_size = blurred_img.shape[0]
            parallel = torch.mean(compare_ftn(start, frame0) + compare_ftn(end, frame1), dim=(1,2,3))
            cross = torch.mean(compare_ftn(start, frame1) + compare_ftn(end, frame0), dim=(1,2,3))

            I0 = torch.zeros_like(blurred_img)
            I1 = center
            
            psnrs = np.zeros((batch_size, seq_len))
            ssims = np.zeros((batch_size, seq_len))

            for vindex in range(seq_len):                    
                frameT = validationData[vindex]
                IFrame = frameT.to(device)
                
                if vindex == 0:
                    Ft_p = torch.zeros_like(blurred_img)
                    for b in range(batch_size):
                        if parallel[b] <= cross[b]:
                            Ft_p[b] = start[b].clone()
                        else:
                            Ft_p[b] = end[b].clone()

                elif vindex == seq_len-1:
                    Ft_p = torch.zeros_like(blurred_img)
                    for b in range(batch_size):
                        if parallel[b] <= cross[b]:
                            Ft_p[b] = end[b].clone()
                        else:
                            Ft_p[b] = start[b].clone()

                elif vindex == ctr_idx:
                    Ft_p = center.clone()

                else:
                    validationIndex = torch.ones(batch_size) * (vindex - 1)
                    validationIndex = validationIndex.long()
                    if vindex > ctr_idx:
                        validationIndex = seq_len - 3 - validationIndex

                    for b in range(batch_size):
                        if (vindex < ctr_idx and parallel[b] <= cross[b]) or (vindex > ctr_idx and parallel[b] > cross[b]):
                            I0[b] = start[b]
                        else:
                            I0[b] = end[b]


                    flowOut = flowComp(torch.cat((I0, I1), dim=1))
                    F_0_1 = flowOut[:,:2,:,:]
                    F_1_0 = flowOut[:,2:,:,:]

                    fCoeff = superslomo.getFlowCoeff(validationIndex, device, seq_len)

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
                    
                    if args.add_blur:
                        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
                    else:
                        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0
                    
                    g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
                    
                    wCoeff = superslomo.getWarpCoeff(validationIndex, device, seq_len)
                    
                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                Ft_p = meanshift(Ft_p, mean, std, device, False)
                IFrame = meanshift(IFrame, mean, std, device, False)

                out = quantize(Ft_p[b])
                foldername = os.path.basename(os.path.dirname(validationFile[ctr_idx][b]))
                if not os.path.exists(os.path.join(imgsave_folder, foldername)):
                    os.makedirs(os.path.join(imgsave_folder, foldername))
                for b in range(batch_size):
                    filename = os.path.splitext(os.path.basename(validationFile[vindex][b]))[0]
                    
                    out_fname = foldername + '_' + filename + '_out.png'
                    # gt_fname = foldername + '_' + filename + '.png'

                    # Comment two lines below if you want to save images
                    torchvision.utils.save_image(out, os.path.join(imgsave_folder, foldername, out_fname), normalize=True, range=(0,255))
                    # torchvision.utils.save_image(gt, os.path.join(imgsave_folder, gt_fname), normalize=True, range=(0,255))

                psnr, ssim = eval_metrics(Ft_p, IFrame)
                psnrs[:, vindex] = psnr.cpu().numpy()
                ssims[:, vindex] = ssim.cpu().numpy()

            for b in range(batch_size):
                rows = [validationFile[ctr_idx][b]]
                rows.extend(list(psnrs[b]))
                df = df.append(pd.Series(rows, index=df.columns), ignore_index=True)
            
            df.to_csv('{}/results_PSNR.csv'.format(args.checkpoint_dir))


### Initialization
if args.train_continue or args.test_only:
    dict1 = torch.load(args.checkpoint)
    checkpoint_counter = dict1['epoch'] + 1
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
    print()
else:
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}


if args.test_only:
    print("Test Start")
    test()
    print("Test End")
    sys.exit(0)

### Training
import time

starttime = time.time()
cLoss   = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = 0

### Main training loop
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)
        
    # Append and reset
    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    iLoss = 0

    tqdm_trainloader = tqdm.tqdm(trainloader, ncols=80)

    for trainIndex, (trainData, trainFrameIndex, _) in enumerate(tqdm_trainloader):
        ## Getting the input and the target from the training set
        # frame0, frameT, frame1 = trainData
        blurred_img = torch.zeros_like(trainData[0])
        for image in trainData:
            blurred_img += image
        blurred_img /= len(trainData)
        blurred_img = blurred_img.to(device)
        
        with torch.no_grad():
            blurred_img = meanshift(blurred_img, mean, std, device, False)
            center = center_estimation(blurred_img)
            start, end = border_estimation(blurred_img, center)
            start = meanshift(start, mean, std, device, True)
            end = meanshift(end, mean, std, device, True)
            center = meanshift(center, mean, std, device, True)
            blurred_img = meanshift(blurred_img, mean, std, device, True)

        frame0 = trainData[0].to(device)
        frame1 = trainData[-1].to(device)

        batch_size = blurred_img.shape[0]
        parallel = torch.mean(compare_ftn(start, frame0) + compare_ftn(end, frame1), dim=(1,2,3))
        cross = torch.mean(compare_ftn(start, frame1) + compare_ftn(end, frame0), dim=(1,2,3))

        # I0 = torch.zeros_like(blurred_img)
        IFrame = torch.zeros_like(blurred_img)
        choose_start = torch.zeros(batch_size).to(device)

        for b in range(batch_size):
            if (trainFrameIndex[b] < (ctr_idx - 1) and parallel[b] <= cross[b]) or (trainFrameIndex[b] > (ctr_idx - 1) and parallel[b] > cross[b]):
                # I0[b] = start[b]
                choose_start[b] = 1
            else:
                # I0[b] = end[b]
                choose_start[b] = 0

            IFrame[b] = trainData[trainFrameIndex[b]+1][b]
            if trainFrameIndex[b] > (ctr_idx - 1):
                trainFrameIndex[b] = seq_len - 3 - trainFrameIndex[b]

        choose_start = choose_start.reshape((-1, 1, 1, 1))
        I0 = choose_start * start + (1-choose_start) * end
        I1 = center

        optimizer.zero_grad()

        if args.amp:
            with torch.cuda.amp.autocast():
                # Calculate flow between reference frames I0 and I1
                flowOut = flowComp(torch.cat((I0, I1), dim=1))
                
                # Extracting flows between I0 and I1 - F_0_1 and F_1_0
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]
                
                fCoeff = superslomo.getFlowCoeff(trainFrameIndex, device, seq_len)
                
                # Calculate intermediate flows
                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                
                # Get intermediate frames from the intermediate flows
                g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)
                
                # Calculate optical flow residuals and visibility maps
                if args.add_blur:
                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
                else:
                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                # Extract optical flow residuals and visibility maps
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                
                # Get intermediate frames from the intermediate flows
                g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
                
                wCoeff = superslomo.getWarpCoeff(trainFrameIndex, device, seq_len)
                
                # Calculate final intermediate frame 
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                # Loss
                recnLoss = L1_lossFn(Ft_p, IFrame)
                    
                prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
                
                warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)
                
                loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
                loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
                loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
                
                # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
                # since the loss in paper is calculated for input pixels in range 0-255
                # and the input to our network is in range 0-1
                loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
        
            # Backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            # Calculate flow between reference frames I0 and I1
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            
            # Extracting flows between I0 and I1 - F_0_1 and F_1_0
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]
            
            fCoeff = superslomo.getFlowCoeff(trainFrameIndex, device, seq_len)
            
            # Calculate intermediate flows
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
            
            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)
            
            # Calculate optical flow residuals and visibility maps
            if args.add_blur:
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
            else:
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
            
            # Extract optical flow residuals and visibility maps
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
            
            # Get intermediate frames from the intermediate flows
            g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
            
            wCoeff = superslomo.getWarpCoeff(trainFrameIndex, device, seq_len)
            
            # Calculate final intermediate frame 
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            
            # Loss
            recnLoss = L1_lossFn(Ft_p, IFrame)
                
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)
            
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
            # since the loss in paper is calculated for input pixels in range 0-255
            # and the input to our network is in range 0-1
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
    
            # Backpropagate
            loss.backward()
            optimizer.step()

        iLoss += loss.item()

    # Validation and progress every `args.progress_iter` iterations
    # if (trainIndex % args.progress_iter) == args.progress_iter - 1:
    endtime = time.time()
    
    psnr, vLoss, valImg = validate()
    
    valPSNR[epoch].append(psnr)
    valLoss[epoch].append(vLoss)
    
    #Tensorboard
    itr = trainIndex + epoch * len(trainloader)
    
    writer.add_scalars('Loss', {'trainLoss': iLoss/len(trainloader),
                                'validationLoss': vLoss}, itr)
    writer.add_scalar('PSNR', psnr, itr)
    writer.add_image('Validation', valImg, itr)
    #####
    
    endVal = time.time()
    
    print("Loss: %0.6f TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LR: %f" % (iLoss / len(trainloader),  endtime - starttime, vLoss, psnr, endVal - endtime, get_lr(optimizer)))
        
    # Increment scheduler count    
    scheduler.step(psnr)
    
    # Create checkpoint after every `args.checkpoint_epoch` epochs
    if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
        dict1 = {
                'Detail':"End to end Super SloMo.",
                'epoch':epoch,
                'timestamp':datetime.datetime.now(),
                'trainBatchSz':args.train_batch_size,
                'validationBatchSz':args.validation_batch_size,
                'learningRate':get_lr(optimizer),
                'loss':cLoss,
                'valLoss':valLoss,
                'valPSNR':valPSNR,
                'state_dictFC': flowComp.state_dict(),
                'state_dictAT': ArbTimeFlowIntrp.state_dict(),
                }
        torch.save(dict1, args.checkpoint_dir + "/SuperSloMo" + str(checkpoint_counter) + ".ckpt")
        checkpoint_counter += 1
