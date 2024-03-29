import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import time
import glob
import numpy as np
from audio_features import wav2fbank

if torch.cuda.is_available():
    device = torch.device('cuda')  # 选择可用的 GPU 设备
    gpu_name = torch.cuda.get_device_name(device)
    print("GPU is:", gpu_name)
else:
    device = torch.device('cpu')
    print("GPU is Not found")

def build_dataset(opt):
    if opt.dataset_name == 'mine':
        dataset = MineDataset(opt.dataset_path, opt.batch_size)
    elif opt.dataset_name == 'audioset':
        dataset = None
    elif opt.dataset_name == 'fdk50k':
        dataset = None
    elif opt.dataset_name == 'esc50':
        dataset = None
    else:
        dataset = None

    train_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size, 
                                                 shuffle=True)
    return train_loader

def train(opt, model, train_loader):
    model.to(device)
    model.train()

    # 初始化模型、损失函数和优化器
    if opt.loss == 'MSE' :
        criterion = nn.MSELoss()
    elif opt.loss == 'BCE' :
        criterion = nn.BCELoss()
    elif opt.loss == 'CE':
        criterion = nn.CrossEntropyLoss()

    if opt.optimizer == 'SGD' :
        optimizer = optim.SGD(model.parameters(), lr = opt.learning_rate)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate)

    time_start = time.perf_counter()
    # 训练模型
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        epoch_time_start = time.perf_counter()
        for step, data  in enumerate(train_loader, start = 0):
            inputs,labels  = data

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % opt.num_steps == opt.num_steps - 1:
                print('epoch: [%d/%d, %5d time: %f s] loss: %.3f' % (epoch + 1, opt.num_epochs,
                 step + 1, time.perf_counter() - epoch_time_start, running_loss / opt.num_steps))
                running_loss = 0.0

    print('epcosh use total: %f s finished training' %(time.perf_counter() - time_start, ))

    torch.save(model.state_dict(), opt.save_path + opt.model_name + '.pth')
    print('save weight: ', opt.save_path + opt.model_name + '.pth')

    if opt.export_onnx:
        input_data = torch.randn(1, 998, 128) 
        torch.onnx.export(model, input_data.to(device),opt.onnx_path + opt.model_name + '.onnx', 
                        input_names=['inputs'], output_names=['outputs'])
        print("export onnx ", opt.onnx_path, "end") 


def test(model, test_loader):
    scores = []
    # 在验证集上评估模型
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_correct = 0
        val_total = 0
        for data in test_loader:
            input_datas, labels = data
            labels.to(device)
            outputs = model(input_datas.to(device))
            _, predicted = torch.max(outputs.data, 1)
            val_loss += criterion(outputs, labels).item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        print('Accuracy on test val: %.2f loss: %.4f%' % (val_accuracy, val_loss))


if __name__ == '__main__':
    print('test')

