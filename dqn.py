import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # First layer
        self.fc2 = nn.Linear(128, 128)         # Second layer
        self.fc3 = nn.Linear(128, action_size) # Output layer for action values

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action
