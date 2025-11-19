import argparse
import random
import sys
from typing import Tuple

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from ml_denoiser.denoiser.models import UNet6Residual, UNetResidual
from ml_denoiser.denoiser.training import fit, evaluate
from ml_denoiser.denoiser.dataset import DenoiserPatchDataset, load_image_tensor, save_image
from ml_denoiser.denoiser.models import UNetResidual

import json

data = {
    "loss": 0.123,
    "epoch": 15,
    "params": {
        "batch_size": 8,
        "lr": 1e-4
    }
}

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)