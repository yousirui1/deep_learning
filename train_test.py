import sys
sys.path.append('/home/ysr/project/ai/deep_learning/torch/model')
import torch
import torchaudio
from torchaudio.transforms import Vol
from torchinfo import summary
from efficient import EffNetAttention    
from mobilenet import MobileNetV2
import time
import torch.nn as nn
import torch.optim as optim
from utils.audioset_dataset import *
import torch.nn.functional as F
import torchvision.ops as ops


if torch.cuda.is_available():
    device = torch.device('cuda')  # 选择可用的 GPU 设备
    gpu_name = torch.cuda.get_device_name(device)
    print("GPU is:", gpu_name)
else:
    device = torch.device('cpu')
    print("GPU is Not found")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算权重
        if self.alpha is not None:
            weights = torch.where(targets == 1, self.alpha, 1.0)
        else:
            weights = 1.0
        
        # 计算Focal Loss
        focal_loss = torch.pow(1 - torch.exp(-ce_loss), self.gamma) * ce_loss * weights
        
        # 求平均损失
        loss = torch.mean(focal_loss)
        
        return loss

class FocalLoss1(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, alpha = self.alpha, gamma = self.gamma)
        # 求平均损失
        loss = torch.mean(focal_loss)
        
        return loss


def train(model, train_loader, batch_size):
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    #criterion = nn.KLDivLoss(reduction='batchmean');
    criterion = FocalLoss(gamma=4)
    #criterion = FocalLoss1(gamma=5, alpha = 1)
    #criterion = focal_loss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)   # 0.001 1e-4
    #num_epochs = 20
    
    model.train()
    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        time_start = time.perf_counter()
        step_time_start = time.perf_counter()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            #print(outputs)
            #print(labels)
            loss = criterion(outputs, labels.to(device))
            #print('loss: ', loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d/%d, %5d time: %f s] loss: %.3f' % (epoch + 1,num_epochs, i + 1, time.perf_counter() - step_time_start, running_loss / 1000))

                running_loss = 0.0
        if epoch % 10 == 9: 
            torch.save(model.state_dict(), 'output/efficient-b2-sigmoid-16-epoch-' + str(epoch) + '.pth')
    print('Finished training')
    torch.save(model.state_dict(), 'output/efficient-b2-sigmoid-16-epoch-last.pth')
    print('save model weight: ',  'output/efficient-b2-sigmoid-16-epoch-last.pth')
        
def test(model, test_loader, batch_size):
    # 在测试集上测试模型
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    print('Accuracy on test : %.2f %%' % (100 * correct / total))

def transfer_weight(batch_size, label_dim, input_shape):
    #model = EffNetAttention(input_shape = (1, 998, 128), label_dim=200, b=2, pretrain=False, head_num=0, activation = 'sigmoid')    #model = EnsembleModel(input_shape, hidden_dims)
    #model.load_state_dict(torch.load(weight_path))

    audio_model = EffNetAttention(input_shape = (1, 998, 128), label_dim=200, b=2, pretrain=False, head_num=4, activation = 'base')
    audio_model.load_state_dict(torch.load('/home/ysr/project/ai/open_source/psla/pretrained_models/fsd_mdl_best_single.pth'), strict=False)

    model = torch.nn.DataParallel(audio_model)  # 使用 DataParallel 进行模型并行处理

    #new_state_dict = {}
    #state_dict = model.state_dict()
    #for name,param in state_dict.items():
    #    if not name.startswith('attention'):
    #        new_state_dict[name] = param

    new_model = EffNetAttention(input_shape = (batch_size, input_shape[0], input_shape[1]),label_dim = label_dim, b=2, pretrain=False, head_num=0, activation = 'softmax')

    #for name,param in new_model.state_dict().items():
    #    if name.startswith('attention'):
    #        new_state_dict[name] = param

    #for name, param in new_model.named_parameters():
    #    if not name.startswith('attention'):
    #        param.requires_grad = False
    # 遍历两个模型的状态字典，比较层名称并赋值
    for name1, param1 in model.named_parameters():
        for name2, param2 in new_model.named_parameters():
            if name1 == name2 and not name1.startswith('attention'):
                #print(name1)
                param2.data = param1.data
                #param2.requires_grad = False
                break  # 找到匹配的层后，跳出内层循环

    return new_model

if __name__ == '__main__':
    dataset_path = '/home/ysr/project/ai/open_source/psla/egs/fsd50k/' 
    audio_conf = {'num_mel_bins': 128, 'target_length': 3000, 'freqm': 48, 
              'timem': 192, 'mixup': 0.5, 'dataset': 'fsd50k', 'mode': 'train',
              'mean': -4.6476, 'std': 4.5699,
              'noise': False}

    dataset_json_file = dataset_path + './datafiles/fsd50k_tr_full.json'
    label_csv = dataset_path + './class_labels_indices.csv'
    batch_size = 8
    input_shape = (3000, 128)
    label_dim = 200 
    num_epochs = 50
    dataset = AudiosetDataset(dataset_json_file, label_csv=label_csv, audio_conf=audio_conf)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)
    model = transfer_weight(batch_size, label_dim, input_shape)
    #model = EffNetAttention(input_shape = (batch_size, input_shape[0], input_shape[1]),label_dim = label_dim, b=2, pretrain=True, head_num=0, activation = 'sigmoid')
    model.to(device)
    summary(model, input_size=(batch_size, input_shape[0], input_shape[1]))
    train(model, train_loader, batch_size)
