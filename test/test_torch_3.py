import torch
import torch.nn as nn
import torch.onnx
import torchvision.models as models

# 创建EfficientNet模型
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.model(x)

# 创建随机输入数据
input_shape = (3, 224, 224)
input_data = torch.randn(1, *input_shape)

# 创建EfficientNet模型实例
num_classes = 1000  # 替换为实际的类别数量
model = EfficientNet(num_classes)

# 加载预训练权重（如果有）
# model.load_state_dict(torch.load('efficientnet_weights.pth'))

# 导出模型为ONNX格式
torch.onnx.export(model, input_data, 'efficientnet.onnx', opset_version=11)

print("ONNX模型导出完成。")

