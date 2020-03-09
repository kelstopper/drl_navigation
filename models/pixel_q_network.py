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
            nn.Conv2d(3, state_size, kernel_size=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(state_size, state_size * 2, kernel_size=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(state_size * 2, state_size * 4, kernel_size=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(state_size * 4, state_size * 8, kernel_size=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(state_size * 8, state_size * 16, kernel_size=2),
            nn.LeakyReLU(0.02),
        )

        self.fc = nn.Sequential(
            nn.Linear(299568, 149784),
            nn.ReLU(),
            nn.Linear(149784, 74892),
            nn.ReLU(),
            nn.Linear(74892, 37446),
            nn.ReLU(),
            nn.Linear(18723, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.cnn(state)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x