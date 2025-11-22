import argparse
import random
import sys
import re
from typing import Tuple

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from ml_denoiser.training.models import UNet6Residual, UNetResidual
import json

def get_frame_number(path: Path) -> str | None:
    match = re.search(r'\.(\d+)', path.stem)
    if match and match.group(1).isdigit():
        return match.group(1)
    else:
        return None

def get_clean_basename(path: Path) -> str:
    return re.sub(r"\.\d+.*$", "", path.stem)

def get_model_name_and_channels(model: nn.Module):

    # Get base architecture name
    base_name = model.__class__.__name__

    # Find the first Conv2d layer in the model
    in_channels = None
    out_channels = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
            base_channels = layer.out_channels
            break

    if in_channels is None or out_channels is None:
        raise ValueError("No Conv2d layer found in the model.")

    return base_name, in_channels, out_channels

def save_training_parameters(
        path: Path,
        model_name: str,
        in_channels: int,
        out_channels: int,
        epochs: int,
        batch_size: int,
        patches_per_image: int,
        patch_size: int,
        n_first_samples: int,
        n_first_frames: int,
        lr: float,
) -> None:
    """Save JSON training parameters."""
    # name, in_channels, out_channels = get_model_name_and_channels(model)
    training_parameters = {
        "model_name": model_name,
        "out_channels": out_channels,
        "in_channels": in_channels,
        "epochs": epochs,
        "batch_size": batch_size,
        "patches_per_image": patches_per_image,
        "patch_size": patch_size,
        "n_first_samples": n_first_samples,
        "n_first_frames": n_first_frames,
        "lr": lr,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(training_parameters, f, indent=4, ensure_ascii=False)