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
from config import I, D, batch_size, learning_rate, hidden_size, num_classes, num_epochs, input_size


save_file = f'./result/PID_params/mnist_PID I={I} D={D} lr={learning_rate}.txt'


if __name__ == '__main__':
    logger = Logger(save_file, title='mnist')
    logger.set_names(['  L__R\t', '  TLoss\t', '  VLoss\t', 'TAcc\t', '  VAcc\t'])

    train_dataset = dsets.MNIST(root='../PIDoptimizer/data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='../PIDoptimizer/data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = F.relu(out)

            out = self.fc2(out)
            out = F.relu(out)

            out = self.fc3(out)

            return out


    net = Net(input_size, hidden_size, num_classes)

    net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()

    # 获取模型参数名称与层名称的对应关系
    # param_names_to_layers = {param_name: module_name for param_name, module_name in net.named_parameters()}

    optimizer = PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    optimizer.InitWeightsWith_Lr(net)

    for epoch in range(num_epochs):
        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28).cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
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
            images = Variable(images.view(-1, 28 * 28)).cuda()
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
