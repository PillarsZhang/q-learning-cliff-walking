import torch
import numpy as np
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple, hidden_dim:int=128):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(np.prod(input_shape), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, np.prod(output_shape))
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = torch.reshape(x, (x.shape[0], *self.output_shape))
        return x


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
        x = torch.movedim(x, 3, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.reshape(x, (x.shape[0], *self.output_shape))
        return x