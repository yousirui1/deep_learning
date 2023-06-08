import torch
import torch.nn as nn
import torch.onnx as onnx

# 定义CRNN模型
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn = nn.GRU(input_size=1024, hidden_size=256, num_layers=2, bidirectional=True)
    
    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels * height, width)  # 将特征图展开成序列
        x = x.permute(2, 0, 1)  # 调整维度顺序以适应RNN输入
        _, hidden = self.rnn(x)
        return hidden

# 创建CRNN模型实例
model = CRNN()

# 随机生成输入数据
batch_size = 1
channels = 1
height = 32
width = 128
input_data = torch.randn(batch_size, channels, height, width)

# 导出模型为ONNX
torch.onnx.export(model, input_data, "crnn.onnx", opset_version=11)

print("ONNX模型已导出：crnn.onnx")

