import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary

class MADE(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MADE, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.masks = self.create_masks()
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            fc_layer = nn.Linear(self.masks[i].sum(), hidden_sizes[i])
            self.fc_layers.append(fc_layer)
        
        self.output_layer = nn.Linear(self.masks[-1].sum(), output_size)

    def create_masks(self):
        masks = []
        input_mask = torch.arange(self.input_size)
        masks.append(input_mask)
        for hidden_size in self.hidden_sizes:
            hidden_mask = torch.randint(low=0, high=self.input_size, size=(hidden_size,))
            masks.append(hidden_mask)
        output_mask = input_mask
        masks.append(output_mask)
        return masks

    def forward(self, x):
        h = x
        for i, fc_layer in enumerate(self.fc_layers):
            h = h[:, self.masks[i]]
            h = fc_layer(h)
            h = F.relu(h)
        
        output = h[:, self.masks[-1]]
        output = self.output_layer(output)
        return output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义 GroupMADE 模型
class GroupMADE(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_frames):
        super(GroupMADE, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_frames = num_frames
        self.made = MADE(input_size * num_frames, hidden_sizes, output_size * num_frames * 3)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1).expand(-1, self.num_frames, -1)
        x = x.contiguous().view(batch_size * self.num_frames, -1)
        output = self.made(x)
        output = output.view(batch_size, self.num_frames, -1)
        return output

# 定义随机数据生成器
def generate_random_data(batch_size, num_frames, num_mel_bins):
    return torch.randn(batch_size, num_frames, num_mel_bins)

# 定义训练函数
def train(model, optimizer, criterion, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            inputs = data  # 获取输入数据
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, inputs)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

# 定义验证函数
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs = data  # 获取输入数据
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, inputs)  # 计算损失
            total_loss += loss.item() * inputs.size(0)
    average_loss = total_loss / len(val_loader.dataset)
    print(f'Validation Loss: {average_loss}')

# 定义超参数和数据参数
input_size = 128 * 5  # 5 帧，每帧有 128 个 Mel 频带
hidden_sizes = [128, 128, 128, 128, 32, 128, 128, 128, 128]
output_size = 128 * 5 * 3  # 5 帧，每帧有 128 个 Mel 频带，每个混合高斯分量有 3 个参数
num_frames = 5
batch_size = 1
num_epochs = 10

# 创建 GroupMADE 模型
model = GroupMADE(input_size, hidden_sizes, output_size, num_frames)

# 创建随机训练数据和验证数据生成器
train_loader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(generate_random_data(batch_size, num_frames, input_size)),
    batch_size=batch_size,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(generate_random_data(batch_size, num_frames, input_size)),
    batch_size=batch_size,
    shuffle=False
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和验证模型
train(model, optimizer, criterion, train_loader, num_epochs)
validate(model, val_loader)

