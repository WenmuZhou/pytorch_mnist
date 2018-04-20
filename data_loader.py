# -*- coding: utf-8 -*-
# @Time    : 18-4-20 下午3:15
# @Author  : zhoujun
import torch
import torch.utils.data as Data
from PIL import Image
import cv2
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, data_shape,channel=3, transform=None, target_transform=None):
        '''

        :param txt:
        :param data_shape:
        :param channel:
        :param transform:
        :param target_transform:
        '''
        fh = open(txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append((words[0], int(words[1])))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.data_shape = data_shape
        self.channel = channel


    def __getitem__(self, index):
        img_path, label = self.data[index]
        if self.channel == 3:
            img = cv2.imread(img_path,1)
        else:
            img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(img,(self.data_shape[0], self.data_shape[1],self.channel))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
