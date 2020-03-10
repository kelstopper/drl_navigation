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
            nn.Conv2d(3, 84, kernel_size=3, stride=2),
            nn.LeakyReLU(0.02),
            nn.Conv2d(84, 168, kernel_size=3, stride=2),
            nn.BatchNorm2d(168),
            nn.LeakyReLU(0.02),
            nn.Conv2d(168, 336, kernel_size=3, stride=2),
            nn.BatchNorm2d(336),
            nn.LeakyReLU(0.02),
            # nn.Conv2d(336, 672, kernel_size=3, stride=2),
            # nn.BatchNorm2d(672),
            # nn.LeakyReLU(0.02),
            # nn.Conv2d(672, 1344, kernel_size=3, stride=2),
            # nn.BatchNorm2d(1344),
            # nn.LeakyReLU(0.02),
        )

        self.fc = nn.Sequential(
            # nn.Linear(1344, 1344),
            # nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.cnn(state)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x