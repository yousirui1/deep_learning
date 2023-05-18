import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import onnx

# 随机生成训练数据和标签
def generate_random_data():
    inputs = torch.randn(1, 3,224, 224)
    labels = torch.randint(0, 1000, (100,))
    return inputs, labels

# 创建MobileNet模型
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size = (3, 3), stride=(2, 2), 
                padding=(1, 1), bias = False)
        self.model.fc = nn.Linear(1280, num_classes)

    def forward(self, x): 
        #x = x.unsqueeze(0)
        #x = x.view(1, 
        #x = x.transpose(0, 1)
        return self.model(x)

# 替换最后一层全连接层
#num_ftrs = model.classifier[1].in_features
#model.classifier[1] = nn.Linear(num_ftrs, 1000)  # 修改输出类别数

num_classes = 2

#model.apply(init_weights)
model = MobileNetV2(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
#for epoch in range(2):
#    inputs, labels = generate_random_data()
#    optimizer.zero_grad()
#    outputs = model(inputs)
#    loss = criterion(outputs, labels)
#    loss.backward()
#    optimizer.step()

# 将模型转换为ONNX
dummy_input = torch.randn(32, 3, 3)
onnx_path = "mobilenet.onnx"
torch.onnx.export(model, dummy_input, onnx_path)

print("ONNX模型已导出：", onnx_path)

