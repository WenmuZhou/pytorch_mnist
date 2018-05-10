import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from data_loader import MyDataset
import time
from tensorboardX import SummaryWriter

device = torch.device("cuda:0")
print('training with:', device)
num_epochs = 3
batch_size = 64

# train_data = torchvision.datasets.ImageFolder(root='/data/datasets/mnist/train',
#  transform=transforms.Compose(
#  [transforms.Resize(227), transforms.ToTensor()]))
# 训练时间更短，不知道为何
train_data = MyDataset(txt='/data/datasets/mnist/train.txt', data_shape=(227, 227), channel=3,
                       transform=transforms.ToTensor())
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)

model = torchvision.models.AlexNet(num_classes=10)
# 准备写tensorboard, 必须放在'.to(device)'之前，不然会报错
writer = SummaryWriter()
dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 227, 227))
writer.add_graph(model=model, input_to_model=dummy_input)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
model.train()

for epoch in range(num_epochs):
    scheduler.step()
    train_acc, train_loss = 0., 0.
    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
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

        cur_step = epoch * (train_data.__len__()/batch_size)+i
        writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=cur_step)
        writer.add_scalar(tag='Train/acc', scalar_value=acc, global_step=cur_step)
        # if i % 1 == 0:
        # print('Iteration: {}. Loss: {}. Accuracy: {}'.format(cur_step, loss.item(), acc))
    print('epoch [%d/%d], loss: %.4f, acc: %.4f, time:%0.4f, lr:%s' % (
        epoch + 1, num_epochs, train_loss /
        len(train_data), train_acc / len(train_loader.dataset),
        time.time() - start, str(scheduler.get_lr()[0])))
writer.close()
# torch.save(model, 'AlexNet1.pkl')
