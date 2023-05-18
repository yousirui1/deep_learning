import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义GroupMADE模型
class GroupMADE(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size, num_frames):
        super(GroupMADE, self).__init__()
        self.num_frames = num_frames
        self.made = MADE(input_shape, hidden_sizes, output_size, num_frames)

    def forward(self, x):
        return self.made(x)

# 定义MADE模型
class MADE(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size, num_frames):
        super(MADE, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_frames = num_frames

        # 定义掩码
        self.masks = self.create_masks()

        # 创建网络层
        self.input_layer = nn.Linear(self.input_shape[2], self.hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def create_masks(self):
        masks = []
        sizes = [self.input_shape[2]] + self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            mask = torch.zeros((out_size, in_size))
            for i in range(out_size):
                mask[i, :i] = 1
            masks.append(mask)
        return masks

    def forward(self, x):
        x = x.view(-1, self.input_shape[2])

        # 将输入按照掩码进行变换
        for mask in self.masks:
            x = x * mask
            x = self.input_layer(x)
            x = torch.relu(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)

        output = self.output_layer(x)
        return output

# 创建随机数据集
class RandomDataset(Dataset):
    def __init__(self, size, input_shape, output_size, num_frames):
        self.size = size
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_frames = num_frames

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inputs = torch.randn(self.num_frames, *self.input_shape)
        targets = torch.randn(self.num_frames, self.output_size)
        return inputs, targets

# 定义训练和验证函数
def train(model, optimizer, criterion, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def validate(model, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print("Validation Loss: {:.4f}".format(avg_loss))

# 设置模型和训练参数
input_shape = (10, 1268, 128)
hidden_sizes = [128, 128, 64, 64, 32]
output_size = 64
num_frames = 5
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# 创建数据加载器
train_dataset = RandomDataset(1000, input_shape, output_size, num_frames)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = RandomDataset(200, input_shape, output_size, num_frames)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 创建模型、损失函数和优化器
model = GroupMADE(input_shape, hidden_sizes, output_size, num_frames)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证
train(model, optimizer, criterion, train_loader, num_epochs)
validate(model, val_loader)

