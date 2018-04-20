# -*- coding: utf-8 -*-
# @Time    : 18-4-20 下午3:15
# @Author  : zhoujun
import torch
import torch.utils.data as Data
from PIL import Image
import cv2


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, data_shape, transform=None, target_transform=None):
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

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
