import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import time
import cv2
import numpy as np
from PIL import Image

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class Pytorch_model:
    def __init__(self,model_path,img_shape, gpu_id = None,classes = None):
        self.gpu_id = gpu_id
        self.img_shape = img_shape
        if self.gpu_id is not None and isinstance(self.gpu_id,int):
            self.use_gpu = True
        else:
            self.use_gpu = False


        if not self.use_gpu:
            self.net = torch.load(model_path,map_location=lambda storage, loc: storage.cpu())
        else:
            self.net = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
        self.net.eval()

    def predict(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))

        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = Variable(tensor)

        if self.use_gpu:
            tensor = tensor.cuda(self.gpu_id)
        else:
            tensor = tensor.cpu()

        outputs = self.net(tensor)
        _, prediction_tensor = torch.max(outputs.data, 1)

        if self.use_gpu:
            prediction = prediction_tensor.cpu().numpy()[0]
        else:
            prediction = prediction_tensor.numpy()[0]
        return prediction


if __name__ == '__main__':
    img_path = r'/data/datasets/mnist/mnist_img/test/4/1.jpg'
    model_path = 'resnet18.pkl'

    model = Pytorch_model(model_path,img_shape=[224,224])
    start_cpu = time.time()
    epoch = 1
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
        print('device: cpu, result:%d, time: %.4f' % ( result,time.time()-start) )
    end_cpu = time.time()

    # test gpu speed
    # model1 = Pytorch_model(model_path=model_path,img_shape=[224,224], gpu_id=7)
    # start_gpu = time.time()
    # for _ in range(epoch):
    #     start = time.time()
    #     result = model1.predict(img_path)
    #     print('device: gpu, result:%d, time: %.4f' % ( result,time.time()-start) )
    # end_gpu = time.time()
    print('cpu avg time: %.4f' % ((end_cpu-start_cpu)/epoch))
    # print('gpu avg time: %.4f' % (( end_gpu-start_gpu)/epoch))
