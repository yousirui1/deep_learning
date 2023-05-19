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


def build_dataset(opt, dir_name = 'train', ext = 'wav'):
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(
        dir=opt.dataset_path, dir_name=dir_name, ext=ext))
    file_list = sorted(glob.glob(training_list_path))
    
    train_set = torch.empty(len(file_list), 998, 128)
    i = 0
    print("len(file_list)", len(file_list))
    for file_path in file_list:
        fbank, _ = wav2fbank(998, 128, file_path)
        train_set[i] = fbank
        i += 1

    train_loader = torch.utils.data.DataLoader(torch.tensor(train_set), batch_size = 32, shuffle=False, num_workers=12)
    return train_loader


def train(opt, model, train_loader):
    # 输出模型结构
    summary_flag = 1

    model.to(device)
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

    # 训练模型
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        time_start = time.perf_counter()
        step_time_start = time.perf_counter()

        for step, data  in enumerate(train_loader, start = 0):
            inputs  = data
            if summary_flag == 1:
                print(inputs.shape)
                summary(model, input_size= (1, 998, 128))
                summary_flag = 0;

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs.to(device))
            loss = criterion(inputs, outputs)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 99:
                with torch.no_grad():
                    #for step, data  in enumerate(train_loader, start = 0):
                    #outputs = model(test_image.to(device))
                    #predict_y = torch.max(outputs, dim=1)[1]
                    #accuracy = (predict_y == 
                    #        test_label.to(device)).sum().item() / test_label.size(0)

                    #print('[epoch: %d/%d, step: %5d time: %f s] train_loss: %.3f test_accuracy: %.3f'
                    #    %(epoch + 1, opt.num_epochs,  step + 1, time.perf_counter() - step_time_start, running_loss / 500, accuracy))

                    print('[epoch: %d/%d, step: %5d time: %f s] train_loss: %.3f '
                        %(epoch + 1, opt.num_epochs,  step + 1, time.perf_counter() - step_time_start, running_loss / 500))
                    running_loss = 0.0
                    step_time_start = time.perf_counter()
        print('epcosh use total: %f s' %(time.perf_counter() - time_start, ))
        
    print('end train model')

    torch.save(model.state_dict(), opt.save_path + opt.model_name + '.pth')
    print('save weight: ', opt.save_path)

    if opt.export_onnx:
        torch.onnx.export(model, test_image.to(device),opt.onnx_path + opt.model_name + '.onnx', 
                            input_names=['inputs'], output_names=['outputs'])
        print("export onnx ", opt.onnx_path, "end") 



def test(model, test_loader):
    #scores = []
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
    #with torch.no_grad():
    #    for 

    return None


if __name__ == '__main__':
    build_wav_data('/home/ysr/project/dataset/audio/dcase2020_task2/train/fan/')


