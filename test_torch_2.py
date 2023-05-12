import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import onnx

# 随机生成训练数据和标签
def generate_random_data():
    inputs = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 1000, (100,))
    return inputs, labels

# 创建MobileNet模型
model = models.mobilenet_v2(pretrained=False)

# 替换最后一层全连接层
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 1000)  # 修改输出类别数

# 随机初始化模型参数
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)

model.apply(init_weights)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(2):
    inputs, labels = generate_random_data()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 将模型转换为ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "mobilenet.onnx"
torch.onnx.export(model, dummy_input, onnx_path)

print("ONNX模型已导出：", onnx_path)

