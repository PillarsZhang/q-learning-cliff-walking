from typing import Sequence
import torch
import numpy as np
from pathlib import Path
from torch import nn

class CNN(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple, hidden_dim:int=128):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.conv1 = nn.Conv2d(self.input_shape[2], 16, 3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1]*32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, np.prod(output_shape))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.reshape(x, (x.shape[0], *self.output_shape))
        return x

def calc_same_padding(kernel_size, stride, input_size):
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    input_size = np.array(input_size)
    pad_all = (stride - 1) * input_size - stride + kernel_size
    pad_0 = pad_all // 2
    pad_1 = pad_all - pad_0
    if (pad_0 == pad_1).all():
        if pad_0[0] == pad_0[1]:
            return pad_0[0]
        else:
            return tuple(pad_0)
    else:
        return *tuple(pad_0), *tuple(pad_1)

def replace_conv2d_with_same_padding(m: nn.Module, input_size: tuple):
    if isinstance(m, nn.Conv2d):
        if m.padding == "same":
            m.padding = calc_same_padding(
                kernel_size=m.kernel_size,
                stride=m.stride,
                input_size=input_size
            )

input = torch.rand(1, 4, 12, 4)
model = CNN(input_shape=(4, 12, 4), output_shape=(4,))

model.apply(lambda m: replace_conv2d_with_same_padding(m, input_size=(4,12)))

saved_path = Path("saved/test_onnx_export/")
saved_path.mkdir(exist_ok=True, parents=True)

input_names = ['s_t']
output_names = ['Q(s_t, a)']
onnx_path = saved_path / "advanced_dqn_model.onnx"
torch.onnx.export(model, input, onnx_path, input_names=input_names, output_names=output_names)

# ts_path = saved_path / "advanced_dqn_model.pt"
# m = torch.jit.script(model)
# m.save(ts_path)
