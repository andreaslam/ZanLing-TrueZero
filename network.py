import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TrueNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, head_channel_policy, head_channel_values):
        super().__init__()

        self.startBlock = nn.Sequential(
            nn.Conv2d(21, num_hidden, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_policy, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_policy),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channel_policy * 64, 1880),
        )

        # initialize the weights of the policyHead Linear layer to zeros to achieve flat policy
        self.policyHead[-1].weight.data.fill_(0)
        self.policyHead[-1].bias.data.fill_(0)

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_values, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_values),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channel_values * 64, 5),
            nn.Tanh(),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_values, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_values),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channel_values * 64, 5),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x_temp = block(x)
            x = x_temp + x
            x = F.relu(x)

        policy = self.policyHead(x)

        value = self.valueHead(x)

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
        x = x + residual
        return x
