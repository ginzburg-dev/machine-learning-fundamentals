import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from PIL import Image

class UNet6Residual(nn.Module):
    """Predict NOISE then subtract"""

    def __init__(self, channels=3, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, channels, 3, padding=1),
        )

    def forward(self, noisy):
        x = self.encoder(noisy)
        x = self.middle(x)
        noise_pred = self.decoder(x)
        clean_pred = noisy - noise_pred
        return clean_pred
