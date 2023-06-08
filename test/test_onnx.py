import torch
import torch.onnx as onnx
import torchvision.models as models

# 加载TorchScript模型
model = torch.jit.load("model.pt")

# 示例输入数据
dummy_input = torch.randn(1, 10)

# 导出ONNX模型
onnx_path = "model.onnx"
onnx.export(model, dummy_input, onnx_path)

print("ONNX模型已导出：", onnx_path)
