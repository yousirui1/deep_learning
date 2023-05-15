import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if torch.cuda.is_available():
    device = torch.device('cuda')  # 选择可用的 GPU 设备
    gpu_name = torch.cuda.get_device_name(device)
    print("GPU is:", gpu_name)
else:
    device = torch.device('cpu')
    print("GPU is Not found")

def build_dataset(opt):
    train_set = torchvision.datasets.CIFAR10(root=opt.dataset_path, train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, num_workers=12)

    test_set = torchvision.datasets.CIFAR10(root=opt.dataset_path, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=12)

    for images, labels in test_loader:
        print(images)
        print(images.shape)
        break


    return train_loader, test_loader, 0


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
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(opt, model, input_shape, train_loader, test_loader):

    model = LeNet()

    summary(model, input_size=input_shape)
    model.to(device)
    
    # 初始化模型、损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = opt.learning_rate)

    test_image = None
    test_label = None

    for test_image, test_label in test_loader:
        test_image = test_image
        test_label = test_label
        break

    # 训练模型
    total_step = len(train_loader)
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        time_start = time.perf_counter()
        step_time_start = time.perf_counter()

        for step, data  in enumerate(train_loader, start = 0):
            inputs, labels = data
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

#            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
#                    .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))

            running_loss += loss.item()
            if step % 100 == 99:
                with torch.no_grad():
                    #for step, data  in enumerate(train_loader, start = 0):

                    outputs = model(test_image.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == 
                            test_label.to(device)).sum().item() / test_label.size(0)

                    print('[epoch: %d/%d, step: %5d time: %f s] train_loss: %.3f test_accuracy: %.3f'
                        %(epoch + 1, opt.num_epochs,  step + 1, time.perf_counter() - step_time_start, running_loss / 500, accuracy))
                    running_loss = 0.0
                    step_time_start = time.perf_counter()
        print('epcosh use total: %f s' %(time.perf_counter() - time_start, ))
        
    print('end train model')

    # 在验证集上评估模型
    #model.eval()
    #with torch.no_grad():
    #    val_loss = 0
    #    val_correct = 0
    #    val_total = 0
    #    for images, labels in test_loader:
    #        outputs = model(images)
    #        _, predicted = torch.max(outputs.data, 1)
    #        val_loss += criterion(outputs, labels).item()
    #        val_total += labels.size(0)
    #        val_correct += (predicted == labels).sum().item()

    #    val_loss /= len(test_loader)
    #    val_accuracy = 100 * val_correct / val_total

    #    print('Epoch [{}/{}], Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
    #          .format(epoch + 1, num_epochs, val_loss, val_accuracy))


    torch.save(model.state_dict(), opt.save_path + opt.model_name + '.pth')
    print('save weight: ', opt.save_path)

    if opt.export_onnx:
        torch.onnx.export(model, test_image.to(device), opt.onnx_path + opt.model_name + '.onnx', 
                            input_names=['inputs'], output_names=['outputs'])

        print("export onnx ", opt.onnx_path, "end") 

