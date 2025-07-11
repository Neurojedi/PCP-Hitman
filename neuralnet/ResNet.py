import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game_utils import (
    initialize_game_state, 
    check_end_state, 
    GameState, 
    apply_player_action, 
    PLAYER1, 
    PLAYER2
)
from agents.MCTS import Node, MCTS, random_eval  # Imported here in case of dynamic module context

# The neural network used for evaluating Connect4 board states and suggesting optimal moves.
# It outputs a scalar value estimate (how favorable the position is) and a policy distribution
# over possible next actions, using convolutional layers with residual connections
class Connect4Model(nn.Module):
    """
    A convolutional neural network for Connect4 with residual blocks and dual heads 
    for value and policy prediction.

    The model takes a (3 × 6 × 7) input tensor and outputs:
    - a scalar value estimate (via the value head)
    - a 7-dimensional policy vector (via the policy head)

    Residual connections are used in four stacked residual blocks after the initial convolution,
    each consisting of two convolutional layers with skip connections.

    Note: Batch normalization layers are defined but not consistently used in the forward pass. Bias is not included for most layers.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Initial conv
        self.initial_conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.initial_bn = nn.BatchNorm2d(128)
        
        # Res block 1
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(128)
        
        # Res block 2
        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(128)
        
        # Res block 3
        self.res3_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res3_bn1 = nn.BatchNorm2d(128)
        self.res3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res3_bn2 = nn.BatchNorm2d(128)

        # Res block 4
        self.res4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res4_bn1 = nn.BatchNorm2d(128)
        self.res4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.res4_bn2 = nn.BatchNorm2d(128)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1, bias=True)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc = nn.Linear(3*6*7, 32)
        self.value_head = nn.Linear(32, 1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=True)
        self.policy_bn = nn.BatchNorm2d(32)
        # Change from 32*6*7 to 32 features after global average pooling
        self.policy_head = nn.Linear(32, 7)
        
        self.to(device)
    
    # connecting the layers in the forward method
    def forward(self, x):
        x = x.view(-1, 3, 6, 7)
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Res block 1
        res = x
        x = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = self.res1_bn2(self.res1_conv2(x))
        x = x + res
        x = F.relu(x)
        
        # Res block 2
        res = x
        x = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = self.res2_bn2(self.res2_conv2(x))
        x = x + res
        x = F.relu(x)
        
        # Res block 3
        res = x
        x = F.relu(self.res3_bn1(self.res3_conv1(x)))
        x = self.res3_bn2(self.res3_conv2(x))
        x = x + res
        x = F.relu(x)

        # Res block 4
        res = x
        x = F.relu(self.res4_bn1(self.res4_conv1(x)))
        x = self.res4_bn2(self.res4_conv2(x))
        x = x + res
        x = F.relu(x)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 3*6*7)
        value = F.relu(self.value_fc(value))
        value = self.value_head(value)
        value = torch.tanh(value)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.mean(policy, dim=(2, 3))  # Global average pool → [B, 32]
        policy_logits = self.policy_head(policy)  # → [B, 7]

        return value, policy_logits
