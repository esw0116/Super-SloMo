
#[Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import superslomo
from model.extraction import center, ends
from data import gopro_blur
import dataloader

from copy import deepcopy
from math import log10
import datetime
import os


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--seq_len", type=int, default=11, help='number of frames that composes a sequence.')
parser.add_argument("--train_batch_size", type=int, default=8, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=2, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=1000, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
args = parser.parse_args()

### For visualizing loss and interpolated frames

writer = SummaryWriter('log')
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

###Initialize flow computation and arbitrary-time flow interpolation CNNs.


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = superslomo.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = superslomo.UNet(23, 5)
ArbTimeFlowIntrp.to(device)


###Initialze backward warpers for train and validation datasets

trainFlowBackWarp      = superslomo.backWarp(352, 352, device)
trainFlowBackWarp      = trainFlowBackWarp.to(device)
validationFlowBackWarp = superslomo.backWarp(1280, 704, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)


### Load Pretrained Extraction Models

center_estimation = center.Center()
border_estimation = ends.Ends()

pretrained_weight = torch.load('pretrained_models/best_gopro.ckpt')['state_dict']
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
print('Estimation network')

### Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

# trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

# validationset = dataloader.SuperSloMo(root=args.dataset_root + '/validation', transform=transform, randomCropSize=(640, 352), train=False)
# validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

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

L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

optimizer = optim.Adam(params, lr=args.init_learning_rate)
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)


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
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            # frame0, frameT, frame1 = validationData

            # I0 = frame0.to(device)
            # I1 = frame1.to(device)
            # IFrame = frameT.to(device)

            blurred_img = torch.zeros_like(validationData[0])
            for image in validationData:
                blurred_img += image
            blurred_img /= len(validationData)
            blurred_img = blurred_img.to(device)
            
            c = center_estimation(blurred_img)
            start, end = border_estimation(blurred_img, c)

            compare_ftn = nn.L1Loss()
            frame0 = validationData[0].to(device)
            frame1 = validationData[-1].to(device)
            parallel = True if compare_ftn(start, frame0) + compare_ftn(end, frame1) <= compare_ftn(start, frame1) + compare_ftn(end, frame0) else False

            if parallel:
                I0, I1 = start, end
            else:
                I0, I1 = end, start

            frameT = torch.zeros_like(I0)
            for i, fidx in enumerate(validationFrameIndex):
                frameT[i] = validationData[fidx.item()+1][i]

            IFrame = frameT.to(device)
            
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = superslomo.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
            
            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
                
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
                
            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            
            wCoeff = superslomo.getWarpCoeff(validationFrameIndex, device)
            
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            
            # For tensorboard
            if flag:
                retImg = torchvision.utils.make_grid([revNormalize(frame0.cpu()[0]), revNormalize(frameT.cpu()[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1.cpu()[0])], padding=10)
                flag = 0
            
            #loss
            recnLoss = L1_lossFn(Ft_p, IFrame)
            
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
        
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()
            
            #psnr
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))

            # Make benchmark csv file

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg


### Initialization


if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
else:
    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}


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
    
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        # frame0, frameT, frame1 = trainData
        blurred_img = torch.zeros_like(trainData[0])
        for image in trainData:
            blurred_img += image
        blurred_img /= len(trainData)
        blurred_img = blurred_img.to(device)
        
        with torch.no_grad():
            c = center_estimation(blurred_img)
            start, end = border_estimation(blurred_img, c)

        compare_ftn = nn.L1Loss()
        frame0 = trainData[0].to(device)
        frame1 = trainData[-1].to(device)
        parallel = True if compare_ftn(start, frame0) + compare_ftn(end, frame1) <= compare_ftn(start, frame1) + compare_ftn(end, frame0) else False

        if parallel:
            I0, I1 = start, end
        else:
            I0, I1 = end, start

        # I0 = frame0.to(device)
        # I1 = frame1.to(device)
        IFrame = torch.zeros_like(I0)
        for i, fidx in enumerate(trainFrameIndex):
            IFrame[i] = trainData[fidx.item()+1][i]

        IFrame = IFrame.to(device)
        
        optimizer.zero_grad()
        
        # Calculate flow between reference frames I0 and I1
        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        
        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]
        
        fCoeff = superslomo.getFlowCoeff(trainFrameIndex, device)
        
        # Calculate intermediate flows
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)
        
        # Calculate optical flow residuals and visibility maps
        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0, blurred_img), dim=1))
        
        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
        
        wCoeff = superslomo.getWarpCoeff(trainFrameIndex, device)
        
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
        if (trainIndex % args.progress_iter) == args.progress_iter - 1:
            endtime = time.time()
            
            psnr, vLoss, valImg = validate()
            
            valPSNR[epoch].append(psnr)
            valLoss[epoch].append(vLoss)
            
            #Tensorboard
            itr = trainIndex + epoch * (len(trainloader))
            
            writer.add_scalars('Loss', {'trainLoss': iLoss/args.progress_iter,
                                        'validationLoss': vLoss}, itr)
            writer.add_scalar('PSNR', psnr, itr)
            writer.add_image('Validation',valImg , itr)
            #####
            
            endVal = time.time()
            
            print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (iLoss / args.progress_iter, trainIndex, len(trainloader), endtime - starttime, vLoss, psnr, endVal - endtime, get_lr(optimizer)))
            
            cLoss[epoch].append(iLoss/args.progress_iter)
            iLoss = 0
            starttime = time.time()
        
    # Increment scheduler count    
    scheduler.step()
    
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
