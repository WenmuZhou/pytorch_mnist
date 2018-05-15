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


def evaluate_accuracy(test_loader, net):
    net.eval()
    n = 0
    acc = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        _, preds = torch.max(out.data, 1)
        correct = preds.eq(labels.data).sum().item()
        acc += correct
        n += labels.size(0)
    return acc / n


# train_data = torchvision.datasets.ImageFolder(root='/data/datasets/mnist/train',
#  transform=transforms.Compose(
#  [transforms.Resize(227), transforms.ToTensor()]))
# 训练时间更短，不知道为何
train_data = MyDataset(txt='/data/datasets/mnist/train.txt', data_shape=(227, 227), channel=3,
                       transform=transforms.ToTensor())
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)

test_data = MyDataset(txt='/data/datasets/mnist/test.txt', data_shape=(227, 227), channel=3,
                     transform=transforms.ToTensor())
test_loader = Data.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=3)

net = torchvision.models.AlexNet(num_classes=10)
net = net.to(device)

writer = SummaryWriter('./log/%s' % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 227, 227).to(device))
writer.add_graph(model=net, input_to_model=dummy_input)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)

for epoch in range(num_epochs):
    net.train()
    scheduler.step()
    train_acc, train_loss = 0., 0.
    start = time.time()
    cur_step = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Forward
        optimizer.zero_grad()
        out = net(images)
        # Backward
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(out.data, 1)
        correct = preds.eq(labels.data).sum().item()
        acc = correct / labels.size(0)
        train_acc += correct

        cur_step = epoch * (train_data.__len__() / batch_size) + i
        writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=cur_step)
        writer.add_scalar(tag='Train/acc', scalar_value=acc, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=scheduler.get_lr()[0], global_step=cur_step)
        # if i % 1 == 0:
        # print('Iteration: {}. Loss: {}. Accuracy: {}'.format(cur_step, loss.item(), acc))
    eval_acc = evaluate_accuracy(test_loader, net)
    writer.add_scalar(tag='Eval/acc', scalar_value=eval_acc, global_step=cur_step)
    print('epoch [%d/%d], train_loss: %.4f, train_acc: %.4f, eval_acc: %.4f, time:%0.4f, lr:%s' % (
        epoch + 1, num_epochs, train_loss / len(train_data), train_acc / len(train_loader.dataset), eval_acc,
        time.time() - start, str(scheduler.get_lr()[0])))
writer.close()
# torch.save(model, 'AlexNet1.pkl')
