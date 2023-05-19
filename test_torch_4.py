import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.transformer = Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc = nn.Linear(embedding_dim, input_dim)

    def forward(self, src):
        src = self.linear(src)
        src = src.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        output = self.transformer(src, src)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
        output = self.fc(output)
        output = output.view(-1, output.size(2))  # 调整维度去除批次维度
        return output

# 实例化模型
input_dim = 10
embedding_dim = 256
hidden_dim = 512
num_layers = 4
num_heads = 8

model = TransformerModel(input_dim, embedding_dim, hidden_dim, num_layers, num_heads)

# 创建示例输入张量
dummy_input = torch.randn(1, 5, input_dim)  # 输入形状为 [batch_size, seq_len, input_dim]

# 导出模型为ONNX格式
onnx_path = 'transformer_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
print("模型已成功导出为ONNX格式：", onnx_path)


