import glob
import imageio
import numpy as np
import os
from os import path
import random
import tqdm

import torch
from torch import nn
from torch.utils import data


class RedBase(data.Dataset):
    def __init__(self, root, transform=None, dim=(1280, 720), randomCropSize=(352, 352), seq_len=11, train=True):
        super(RedBase, self).__init__()
        self.seq_len = seq_len
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train

        self.data_root = os.path.join(root, 'REDS120fps')
        self.data_dict = self._scan(train)

        self.img_type = 'bin'
        # Pre-decode png files
        if self.img_type == 'bin':
            for k in tqdm.tqdm(self.data_dict.keys(), ncols=80):
                bin_path = os.path.join(self.data_root, 'bin')
                for idx, v in enumerate(self.data_dict[k]):
                    save_as = v.replace(self.data_root, bin_path)
                    save_as = save_as.replace('.png', '')
                    # If we don't have the binary, make it.
                    if not os.path.isfile(save_as+'.npy'):
                        os.makedirs(os.path.dirname(save_as), exist_ok=True)
                        img = imageio.imread(v)
                        # Bypassing the zip archive error
                        # _, w, c = img.shape
                        # dummy = np.zeros((1,w,c))
                        # img_dummy = np.concatenate((img, dummy), axis=0)
                        # torch.save(img_dummy, save_as)
                        np.save(save_as, img)
                    # Update the dictionary
                    self.data_dict[k][idx] = save_as + '.npy'
        
        self.n_samples = 0
        self.n_sample_list = []
        # when testing, we do not overlap the video sequence (0~6, 7~13, ...)
        if train:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) // self.seq_len
            self.n_sample_list.append(self.n_samples)
        else:
            for k in self.data_dict.keys():
                self.n_sample_list.append(self.n_samples)
                self.n_samples += len(self.data_dict[k]) // self.seq_len
            self.n_sample_list.append(self.n_samples)
            
        print("Sample #:", self.n_sample_list)

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
            dir_train = path.join(self.data_root, 'train/train_orig')
            list_seq = glob.glob(dir_train+'/*')
            data_dict = {
                _make_keys(k): sorted(
                    glob.glob(path.join(k, '*' + '.png'))
                ) for k in list_seq
            }

        else:
            dir_test = path.join(self.data_root, 'val/val_orig')
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
        index *= self.seq_len

        if self.train:
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
            blurfile_dir = os.path.dirname(filepath_list[0]).replace('train_orig', 'train_blur').replace('REDS120fps', 'REDS')
            if self.img_type == 'img':
                blurfile_file = '{:08d}.png'.format(index // 5)
            else:
                blurfile_file = '{:08d}.npy'.format(index // 5)
            blurfile = os.path.join(blurfile_dir, blurfile_file)
            filepath_list.append(blurfile)
        else:
            filepath_list = [self.data_dict[key][i] for i in range(index, index+self.seq_len)]
            blurfile_dir = os.path.dirname(filepath_list[0]).replace('val_orig', 'val_blur').replace('REDS120fps', 'REDS')
            if self.img_type == 'img':
                blurfile_file = '{:08d}.png'.format(index // 5)
            else:
                blurfile_file = '{:08d}.npy'.format(index // 5)
            blurfile = os.path.join(blurfile_dir, blurfile_file)
            filepath_list.append(blurfile)

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
            while True:
                IFrameIndex = random.randint(firstFrame + 1, firstFrame + self.seq_len-2)
                if not IFrameIndex == firstFrame + self.seq_len // 2:
                    break

            if random.randint(0, 1):
                # frameRange = [firstFrame, IFrameIndex, firstFrame + 10]
                frameRange = [i for i in range(firstFrame, firstFrame + self.seq_len)]
                returnIndex = IFrameIndex - firstFrame - 1
            else:
                # frameRange = [firstFrame + 10, IFrameIndex, firstFrame]
                frameRange = [i for i in range(firstFrame + self.seq_len-1, firstFrame - 1, -1)]
                returnIndex = firstFrame - IFrameIndex + self.seq_len-2
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = index % (self.seq_len-2)  + 1
            if IFrameIndex == firstFrame + self.seq_len // 2:
                IFrameIndex = IFrameIndex - 1
            returnIndex = IFrameIndex - 1
            # frameRange = [0, IFrameIndex, 10]
            frameRange = [i for i in range(self.seq_len + 1)]  # +1 for blurry image
            randomFrameFlip = 0

        if self.img_type == 'img':
            fn_read = imageio.imread
        elif self.img_type == 'bin':
            fn_read = np.load
        else:
            raise ValueError('Wrong img type: {}'.format(self.img_type))
               
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            # image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # image = _pil_loader(filepath_list[frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            image = fn_read(filepath_list[frameIndex])
            image = image[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
            if randomFrameFlip:
                image = np.ascontiguousarray(image[:, ::-1])

            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
            
        return sample, returnIndex, filepath_list

    def __len__(self):
        return self.n_samples
