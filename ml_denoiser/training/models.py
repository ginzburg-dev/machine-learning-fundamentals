import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from PIL import Image


def get_model(name: str, in_channel: int, out_channel: int) -> nn.Module:
    if name == "UNet6Residual":
        return  UNet6Residual(channels=in_channel, base=out_channel)
    elif name == "UNetResidual":
        return UNetResidual(channels=in_channel, base=out_channel)
    else:
        msg = f"Unknown model name: {name}"
        raise ValueError(msg)


def get_model_code_name(model: str, in_channels: int, out_channels) -> str:
    if model == "UNet6Residual":
        return f"unet6res_{in_channels}ch_{out_channels}base"
    elif model == "UNetResidual":
        return f"unetres_{in_channels}ch_{out_channels}base"
    else:
        msg = f"Unknown model name: {model}"
        raise ValueError(msg)


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetResidual(nn.Module):
    """Small U-Net that predicts noise and subtracts it."""

    def __init__(self, channels: int = 3, base: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(channels, base)          # 64x64
        self.pool1 = nn.MaxPool2d(2)                   # 32x32

        self.enc2 = ConvBlock(base, base * 2)          # 32x32
        self.pool2 = nn.MaxPool2d(2)                   # 16x16

        # Bottleneck
        self.bottleneck = ConvBlock(base * 2, base * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 32x32
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 64x64
        self.dec1 = ConvBlock(base * 2, base)

        # Output: predict noise
        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(noisy)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        noise_pred = self.out_conv(d1)
        clean_pred = noisy - noise_pred
        return clean_pred
