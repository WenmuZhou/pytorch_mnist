import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from torch.autograd import Variable
from data_loader import MyDataset
import time
import os
import torch.nn as nn

use_gpu = True
if use_gpu:
    gpu_id = "7"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print('train with gpu:', gpu_id)

num_epochs = 3
batch_size = 64

# train_data = torchvision.datasets.ImageFolder(root='/data/datasets/mnist/mnist_img/test',
#                                                  transform=transforms.Compose(
#                                                      [transforms.Resize(224), transforms.ToTensor()]))
# train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

train_data = MyDataset(txt='/data/datasets/mnist/mnist_img/mnist_train_label.txt',data_shape=(224,224),channel=3,
                       transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

model = torchvision.models.resnet50(num_classes=10)
if use_gpu:
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()

for epoch in range(num_epochs):
    train_acc, train_loss = 0., 0.
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        labels = Variable(labels)
        # Forward 
        optimizer.zero_grad()
        out = model(images)
        # Backward   
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, preds = torch.max(out.data, 1)
        correct = preds.eq(labels.data).cpu().sum()
        acc = correct / labels.size(0)
        train_acc += correct
        # if i % 20 == 0:
        # print('Iteration: {}. Loss: {}. Accuracy: {}'.format(i, loss.data[0], acc))
    print('epoch [%d/%d], loss: %.4f, acc: %.4f, time:%0.4f' % (
        epoch + 1, num_epochs, train_loss / len(train_data), train_acc / len(train_loader.dataset),
        time.time() - start))
torch.save(model, 'resnet50.pkl')
