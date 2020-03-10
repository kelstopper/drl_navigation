import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PixelQNetwork, self).__init__()

        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=3, stride=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
            nn.Conv2d(256, 512, kernel_size=3, stride=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.02),
            nn.Conv2d(512, 512, kernel_size=3, stride=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.02),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.cnn(state)
        x = x.view(x.size(0), -1)
        return self.fc(x)