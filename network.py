import torch.nn as nn
import torch.nn.functional as F


class TrueNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(21, num_hidden, kernel_size=8, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9, 1880),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden,1, kernel_size=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        # print(x.shape)
        x = self.startBlock(x)
        # print(x.shape)
        for resBlock in self.backBone:
            x = resBlock(x)
            # print(x.shape)
        # print(x.shape)
        policy = self.policyHead(x)
        policy = policy.squeeze(0)
        # print(policy.shape)
        # print(x.shape)
        value = self.valueHead(x)
        value = value.squeeze(0)
        return policy, value


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
