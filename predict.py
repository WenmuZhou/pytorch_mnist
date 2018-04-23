import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import time
import cv2
import numpy as np


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


class Pytorch_model:
    def __init__(self, model_path, img_shape, gpu_id=None, classes_txt=None):
        self.gpu_id = gpu_id
        self.img_shape = img_shape
        if self.gpu_id is not None and isinstance(self.gpu_id, int):
            self.use_gpu = True
        else:
            self.use_gpu = False

        if not self.use_gpu:
            self.net = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
        else:
            self.net = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
        self.net.eval()

        if classes_txt is not None:
            with open(classes_txt, 'r') as f:
                self.idx2label = dict(line.strip().split(' ') for line in f if line)
        else:
            self.idx2label = None

    def predict(self, image_path, topk=1):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))

        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = Variable(tensor)

        if self.use_gpu:
            tensor = tensor.cuda(self.gpu_id)
        else:
            tensor = tensor.cpu()

        outputs = F.softmax(self.net(tensor))
        result = torch.topk(outputs.data[0], k=topk)
        if self.use_gpu:
            index = result[1].cpu().numpy().tolist()
            prob = result[0].cpu().numpy().tolist()
        else:
            index = result[1].numpy().tolist()
            prob = result[0].numpy().tolist()

        if self.idx2label is not None:
            label = []
            for index in index:
                label.append(self.idx2label[str(index)])
            result = label, prob
        else:
            result = index, prob
        return result


if __name__ == '__main__':
    img_path = r'/data/datasets/mnist/mnist_img/test/4/1.jpg'
    model_path = 'resnet50.pkl'

    model = Pytorch_model(model_path, img_shape=[224, 224], classes_txt='labels.txt')
    start_cpu = time.time()
    epoch = 1
    for _ in range(epoch):
        start = time.time()
        result = model.predict(img_path)
        print('device: cpu, result:%s, time: %.4f' % (str(result), time.time() - start))
    end_cpu = time.time()

    # test gpu speed
    model1 = Pytorch_model(model_path=model_path, img_shape=[224, 224], gpu_id=0, classes_txt='labels.txt')
    start_gpu = time.time()
    for _ in range(epoch):
        start = time.time()
        result = model1.predict(img_path)
        print('device: gpu, result:%s, time: %.4f' % (str(result), time.time() - start))
    end_gpu = time.time()
    print('cpu avg time: %.4f' % ((end_cpu - start_cpu) / epoch))
    print('gpu avg time: %.4f' % ((end_gpu - start_gpu) / epoch))
