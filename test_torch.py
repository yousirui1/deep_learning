import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 随机生成训练数据和标签
def generate_random_data():
    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    return inputs, labels

# 创建模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(2):
    inputs, labels = generate_random_data()
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 将模型转换为TorchScript
dummy_input = torch.randn(1, 10)
scripted_model = torch.jit.trace(model, dummy_input)

# 保存TorchScript模型
scripted_model.save("model.pt")

