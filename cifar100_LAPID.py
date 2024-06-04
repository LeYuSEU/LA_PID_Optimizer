import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from pid import PIDOptimizer

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
from config import momentum, weight_decay, batch_size, num_classes, learning_rate, num_epochs, I, D, learning_rate

save_file = f'./result/PID_dataset/CIFAR100_PID I={I} D={D} lr={learning_rate}.txt'


if __name__ == '__main__':
    logger = Logger(save_file, title='cifar10')
    logger.set_names(['  L__R\t', '  TLoss\t', '  VLoss\t', 'TAcc\t', '  VAcc\t'])

    train_dataset = dsets.CIFAR100(root='../PIDoptimizer/data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.CIFAR100(root='../PIDoptimizer/data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    class Net(nn.Module):
        def __init__(self, num_class=10):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.pool = nn.MaxPool2d(2, 2)
            self.act = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

            self.fc1 = nn.Linear(64 * 6 * 6, 1000)
            self.fc2 = nn.Linear(1000, num_class)

        def forward(self, x):
            x = self.act(self.pool(self.conv1(x)))
            x = self.act(self.pool(self.conv2(x)))  # 100 * 64 * 4 * 4
            x = x.view(x.shape[0], -1)

            out = self.fc1(x)
            out = F.relu(out)
            out = self.fc2(out)

            return out

    net = Net(num_class=100)
    net.cuda()
    net.train()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, I=I, D=D)
    optimizer.InitWeightsWith_Lr(net)

    # Train the Model
    for epoch in range(num_epochs):
        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()

        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.item(), images.size(0))
            train_acc_log.update(prec1[0].cpu(), images.size(0))

        print('Epoch [%d/%d], Loss: %.4f, Acc: %.3f%%' % (epoch + 1, num_epochs, train_loss_log.avg, train_acc_log.avg))

        # Test the Model
        net.eval()
        correct = 0
        loss = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            outputs = net(images)
            test_loss = criterion(outputs, labels)
            val_loss_log.update(test_loss.item(), images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1[0].cpu(), images.size(0))

        logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
        print(f'Accuracy of the network on the 10000 test images: {val_acc_log.avg}%, Loss of the network on the 10000 test images: {val_loss_log.avg:.3f}%.')
        print('-' * 150)

    logger.close()
