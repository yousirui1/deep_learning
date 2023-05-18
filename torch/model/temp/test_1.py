import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__()
        self.mask = mask
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _= nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, in_features):
        masked_weight = self.mask * self.weight
        return nn.functional.linear(in_features, masked_weight, self.bias)


class AutoMaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AutoMaskEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def encoder_layer(self):
        layers = nn.ModuleList()
        in_size = self.input_dim
        for out_size in enumerate(hidden_dims):
            mask = self.create_mask(in_size, input_dim)
            layer = MaskedLinear(in_size, out_size, mask)
            layers.append(layer)
            #self.layers.append(nn.ReLU())
            in_size = out_size
        return layers

    def decoder_layer(self):
        layers = nn.ModuleList()
        for out_size in reversed(hidden_dims):
            in_size = out_size
            mask = self.create_mask(in_size, out_size)
            layer = MaskedLinear(in_size, out_size, mask)
            self.layers.append(layer)
        return layers

    def create_mask(self, in_size, out_size):
        mask = torch.zeros((out_size, in_size))
        for i in range(out_size):
            mask[i, :i] = 1 
        return mask

    def forward(self, x): 
        encoded = self.encoder_layer(x)
        decoded = self.decoder_layer(encoded)
        return decoded

if __name__ ==  '__main__':
    input_dim = 128
    hidden_dims = [128, 128, 64, 64, 32]
    model = AutoMaskEncoder(input_dim, hidden_dims)
    summary(model, input_size=(10, 128))
    ouput = model(test_input)
    onnx_path = "autoencoder.onnx"
    torch.onnx.export(model, test_input, onnx_path)
