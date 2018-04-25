import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from data_loader import MyDataset
import time

device = torch.device("cuda:7")
print('training with:',device)
num_epochs = 3
batch_size = 64

# train_data = torchvision.datasets.ImageFolder(root='/data/datasets/mnist/mnist_img/test',
#                                                  transform=transforms.Compose(
#                                                      [transforms.Resize(224), transforms.ToTensor()]))
#训练时间更短，不知道为何
train_data = MyDataset(txt='/data/datasets/mnist/mnist_img/mnist_train_label.txt',data_shape=(227,227),channel=3,
                       transform=transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

model = torchvision.models.AlexNet(num_classes=10)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()

for epoch in range(num_epochs):
    train_acc, train_loss = 0., 0.
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)
        # Forward
        optimizer.zero_grad()
        out = model(images)
        # Backward   
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(out.data, 1)
        correct = preds.eq(labels.data).sum().item()
        acc = correct / labels.size(0)
        train_acc += correct
        # if i % 20 == 0:
        # print('Iteration: {}. Loss: {}. Accuracy: {}'.format(i, loss.data[0], acc))
    print('epoch [%d/%d], loss: %.4f, acc: %.4f, time:%0.4f' % (
        epoch + 1, num_epochs, train_loss / len(train_data), train_acc / len(train_loader.dataset),
        time.time() - start))
torch.save(model, 'AlexNet.pkl')
