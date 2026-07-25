import torch
import torch.nn.functional as F
from torch import nn

from lib.mapping.mapping import CHESS_FLAT_TO_CONV


class TrueNet(nn.Module):
    def __init__(
        self,
        num_resBlocks=16,
        num_hidden=128,
        head_channel_policy=8,
        head_channel_values=4,
        head_hidden_value=256,
        policy_conv_channels=73,
    ):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(21, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        self.backBoneNorm = nn.BatchNorm2d(num_hidden)

        # AlphaZero-style convolutional policy head.
        # Produces `policy_conv_channels` planes over the 8x8 board, flattened to
        # 73*8*8 = 4672 logits, then gathered down to the 1880 legal-move indices.
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_policy, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_policy),
            nn.ReLU(),
            nn.Conv2d(
                head_channel_policy, policy_conv_channels, kernel_size=1, padding=0
            ),
        )
        # gather indices mapping the 1880-move flat policy into the conv planes
        self.register_buffer("flat_to_conv", CHESS_FLAT_TO_CONV.clone().to(torch.int64))

        # Value head with a hidden fully-connected layer (AlphaZero-style).
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_values, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_values),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channel_values * 8 * 8, head_hidden_value),
            nn.ReLU(),
            nn.Linear(head_hidden_value, 5),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        x = self.backBoneNorm(x)

        policy = self.policyHead(x)  # (b, policy_conv_channels, 8, 8)
        policy = policy.flatten(1)  # (b, 73*8*8)
        # move the gather indices onto the activation's device (cpu/cuda agnostic)
        flat_to_conv = self.flat_to_conv.to(policy.device)
        policy = policy.index_select(1, flat_to_conv)  # (b, 1880)

        value = self.valueHead(x)
        return value, policy


class ResBlock(nn.Module):
    """Pre-activation residual block.

    BN -> ReLU -> conv -> BN -> ReLU -> conv, then add the (unmodified) skip.
    Avoids the double-ReLU / post-activation blow-up of the previous design.
    """

    def __init__(self, num_hidden):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + residual


class TrueNetXS(nn.Module):
    """A much smaller, faster AlphaZero-suitable chess net.

    Same interface and design as `TrueNet` (pre-activation ResNet trunk, AZ-style
    convolutional policy head, value head with a hidden layer) but with far fewer
    residual blocks and channels, so inference is much cheaper. Suitable for
    AlphaZero training/inference; not the toy dense net it replaces.

    Forward returns ``(value, policy)`` with shapes ``(b, 5)`` and ``(b, 1880)``,
    exactly like `TrueNet`.
    """

    def __init__(
        self,
        num_resBlocks=6,
        num_hidden=64,
        head_channel_policy=4,
        head_channel_values=2,
        head_hidden_value=128,
        policy_conv_channels=73,
    ):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(21, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        self.backBoneNorm = nn.BatchNorm2d(num_hidden)

        # AZ-style convolutional policy head (same as TrueNet).
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_policy, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_policy),
            nn.ReLU(),
            nn.Conv2d(
                head_channel_policy, policy_conv_channels, kernel_size=1, padding=0
            ),
        )
        self.register_buffer("flat_to_conv", CHESS_FLAT_TO_CONV.clone().to(torch.int64))

        # Value head with a hidden fully-connected layer.
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, head_channel_values, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_channel_values),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channel_values * 8 * 8, head_hidden_value),
            nn.ReLU(),
            nn.Linear(head_hidden_value, 5),
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        x = self.backBoneNorm(x)

        policy = self.policyHead(x)  # (b, policy_conv_channels, 8, 8)
        policy = policy.flatten(1)  # (b, 73*8*8)
        flat_to_conv = self.flat_to_conv.to(policy.device)
        policy = policy.index_select(1, flat_to_conv)  # (b, 1880)

        value = self.valueHead(x)
        return value, policy
