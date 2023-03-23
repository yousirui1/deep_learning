import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import time

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)           #
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)  # output(32, 5, 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 导入50000 张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform) #预处理
train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=12)

test_set = torchvision.datasets.CIFAR10(root='./data', download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=12)

# 获取测试函数中的标签和图像, 用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = LeNet()
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) 

print('start train model')

for epoch in range(1):
    running_loss = 0.0
    time_start = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data 
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))                   # 正向传播
        loss = loss_function(outputs, labels.to(device))   # 损失计算
        loss.backward()                         # 反向传播
        optimizer.step()

        #print("step, ",step)
        # 打印耗时、损失、准确率等
        running_loss += loss.item()
        if step % 100 == 99:
            with torch.no_grad():       # 不计算每个节点的损失梯度, 防止内存占用
                outputs = net(test_image.to(device))
                
                predict_y = torch.max(outputs, dim=1)[1] # 以output中最大位置对应的索引(标签)作为预测输出
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                print('%f s' %(time.perf_counter() - time_start))
                running_loss = 0.0

print('end train model')

save_path = 'lenet.pth'
torch.save(net.state_dict(), save_path)

torch.onnx.export(net, test_image.to(device), 'lenet.onnx', 
                    input_names=['inputs'], output_names=['outputs'])
