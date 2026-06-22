import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torchvision import transforms

from PIL import Image


def get_loss(name: str) -> _Loss:
    if name == "MSELoss":
        return nn.MSELoss()
    elif name == "L1Loss":
        return nn.L1Loss()
    else:
        msg = f"Unknown loss name: {name}"
        raise ValueError(msg)
