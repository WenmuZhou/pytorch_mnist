# -*- coding: utf-8 -*-
# @Time    : 18-4-20 下午3:15
# @Author  : zhoujun
import torch
import torch.utils.data as Data
import cv2
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, data_shape,channel=3, transform=None, target_transform=None):
        '''

        :param txt:
        :param data_shape:
        :param channel:
        :param transform:
        :param target_transform:
        '''
        with open(txt, 'r') as f:
            data = list(line.strip().split(' ') for line in f if line)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.data_shape = data_shape
        self.channel = channel


    def __getitem__(self, index):
        img_path, label = self.data[index]
        label = int(label)
        img = cv2.imread(img_path, 0 if self.channel==1 else 3)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(img,(self.data_shape[0], self.data_shape[1],self.channel))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
