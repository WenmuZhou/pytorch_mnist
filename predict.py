import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import time
import cv2
import os
import argparse
from shufflenet_v2 import Network


class Pytorch_model:
    def __init__(self, model_path, net, img_shape, img_channel=3, gpu_id=None, classes_txt=None):
        self.img_shape = img_shape
        self.img_channel = img_channel
        self.gpu_id = gpu_id
        self.img_channel = img_channel
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
        else:
            self.device = torch.device("cpu")
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cpu())

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.load_state_dict(self.net)
            self.net = net
        self.net.eval()

        if classes_txt is not None:
            with open(classes_txt, 'r', encoding='utf8') as f:
                self.idx2label = dict(line.strip().split(' ') for line in f if line)
        else:
            self.idx2label = None

    def predict(self, img_path, topk=1):
        assert os.path.exists(img_path) and len(self.img_shape) in [2, 3] and self.img_channel in [1, 3]

        img = cv2.imread(img_path, 0 if self.img_channel == 1 else 1)
        if len(img.shape) == 2 and self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and self.img_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = img.reshape([self.img_shape[0], self.img_shape[1], self.img_channel])
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        outputs = self.net(tensor)
        outputs = F.softmax(outputs, dim=1)
        result = torch.topk(outputs.detach()[0], k=topk)

        if str(self.device) != "cpu":
            index = result[1].cpu().numpy().tolist()
            prob = result[0].cpu().numpy().tolist()
        else:
            index = result[1].numpy().tolist()
            prob = result[0].numpy().tolist()
        if self.idx2label is not None:
            label = []
            for idx in index:
                label.append(self.idx2label[str(idx)])
            result = label, prob
        else:
            result = index, prob
        return result


if __name__ == '__main__':
    img_path = './test.jpg'
    model_path = './model.pth'
    img_shape = [224, 224]
    img_channel = 3
    net = Network(2, 0.25)
    # test cpu speed
    model = Pytorch_model(model_path, net, img_shape=img_shape,
                          img_channel=img_channel, classes_txt='labels.txt')
    start_cpu = time.time()
    epoch = 1000
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
    print('device: cpu, result:%s, time: %.4f' %
              (str(result), time.time() - start))
    end_cpu = time.time()

    # test gpu speed
    model1 = Pytorch_model(model_path=model_path, net=net, img_shape=img_shape,
                           img_channel=img_channel, gpu_id=0, classes_txt='labels.txt')
    start_gpu = time.time()
    for _ in range(epoch):
        start = time.time()
        result = model1.predict(img_path)
    print('device: gpu, result:%s, time: %.4f' %
              (str(result), time.time() - start))
    end_gpu = time.time()
    print('cpu avg time: %.4f' % ((end_cpu - start_cpu) / epoch))
    print('gpu avg time: %.4f' % ((end_gpu - start_gpu) / epoch))
