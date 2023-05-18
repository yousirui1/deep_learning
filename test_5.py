import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# 加载预训练的MobileNetV2模型
model = mobilenet_v2(pretrained=True)

# 定义自定义回归头
regression_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # 全局平均池化层
    nn.Flatten(),  # 展平张量
    nn.Linear(1280, 1)  # 全连接层
)

# 将预训练的MobileNetV2模型的分类头替换为自定义回归头
model.classifier[1] = regression_head

# 定义均方误差损失函数
criterion = nn.MSELoss()

# 生成随机输入和目标张量
batch_size = 10
input_tensor = torch.randn(batch_size, 3, 224, 224)  # 输入张量大小为(batch_size, channels, height, width)
target_tensor = torch.randn(batch_size, 1)  # 目标张量大小为(batch_size, 1)

# 前向传播
output = model(input_tensor)
loss = criterion(output, target_tensor)

print("MSE Loss:", loss.item())

