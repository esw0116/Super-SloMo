import glob
import numpy as np
import os
from os import path
import random

from dataloader import _pil_loader

import torch
from torch import nn
from torch.utils import data


class GoPro(data.Dataset):
    def __init__(self, root, transform=None, dim=(640, 360), randomCropSize=(352, 352), train=True):
        super(GoPro, self).__init__()
        self.seq_len = 11
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train

        self._set_directory(root)
        self.data_dict = self._scan(train)
        
        self.n_samples = 0
        self.n_sample_list = []
        # when testing, we do not overlap the video sequence (0~6, 7~13, ...)
        if train:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) - (self.seq_len - 1)
            self.n_sample_list.append(self.n_samples)
        else:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) // self.seq_len
            self.n_sample_list.append(self.n_samples)
            
        print("Sample #:", self.n_sample_list)

    def _set_directory(self, data_root):
        self.data_root = path.join(data_root, 'GoPro')

    def _scan(self, train):
        def _make_keys(dir_path):
            """
            :param dir_path: Path
            :return: train_000 form
            """
            dir, base = path.dirname(dir_path), path.basename(dir_path)
            tv = 'train' if dir.find('train')>=0 else 'test'
            return tv + '_' + base


        if train:
            dir_train = path.join(self.data_root, 'train')
            list_seq = glob.glob(dir_train+'/*')
            data_dict = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + '.png'))
                ) for k in list_seq
            }

        else:
            dir_test = path.join(self.data_root, 'test')
            list_seq = glob.glob(dir_test+'/*')
            data_dict = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + '.png'))
                ) for k in list_seq
            }

        return data_dict

    def _find_key(self, idx):
        for i, k in enumerate(self.data_dict.keys()):
            if self.n_sample_list[i] <= idx and idx < self.n_sample_list[i+1]:
                return k, idx - self.n_sample_list[i]
        
        raise ValueError()

    def __getitem__(self, idx):
        key, index = self._find_key(idx)
        if self.train:
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
        else:
            index *= self.seq_len
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
        if self.train:
            r = random.random()
            if r > 0.5:
                filepath_list.reverse()

        sample = []
        
        if self.train:
            ### Data Augmentation ###
            # 11 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            # frameRange = range(firstFrame, firstFrame + 9) if (random.randint(0, 1)) else range(firstFrame + 8, firstFrame - 1, -1)
            IFrameIndex = random.randint(firstFrame + 1, firstFrame + 9)
            if random.randint(0, 1):
                frameRange = [firstFrame, IFrameIndex, firstFrame + 10]
                returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [firstFrame + 10, IFrameIndex, firstFrame]
                returnIndex = firstFrame - IFrameIndex + 9
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = ((index) % 9  + 1)
            returnIndex = IFrameIndex - 1
            frameRange = [0, IFrameIndex, 10]
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            # image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            image = _pil_loader(filepath_list[frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            
        return sample, returnIndex

    def __len__(self):
        return self.n_samples
