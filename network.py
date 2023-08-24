import torch.nn as nn
import torch.nn.functional as F


class TrueNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(21, num_hidden, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.Sequential(
            *[ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_hidden * 64, 1880),  # input w*h
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1000, kernel_size=1, padding=0),
            nn.BatchNorm2d(1000),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                1000 * 64, 5
            ), 
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        policy = self.policyHead(x)
        policy = policy.squeeze(0)
        value = self.valueHead(x)
        value = value.squeeze(0)
        return value, policy


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
