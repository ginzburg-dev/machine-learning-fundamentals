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

from pytorch.denoiser.models import UNet6Residual
from pytorch.denoiser.training import fit
from pytorch.denoiser.dataset import DenoiserPatchDataset, load_image_tensor, save_image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "apply"], required=True)

    # common
    p.add_argument("--input", required=True, help="input (DIR - for train, FILE - for apply)")
    p.add_argument("--output", required=True, help="denoised output")

    # for training
    #p.add_argument("--target", help="clean image (for training)")
    p.add_argument("--weights-out", default="denoiser.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patches-per-image", type=int, default=100)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    # for apply
    p.add_argument("--weights-in", default="denoiser.pt")

    return p.parse_args()


def train_mode(args, device):
    #if not args.target:
    #    raise ValueError("--target (clean image) is required in train mode")

    dataset = DenoiserPatchDataset(
        args.input,
        "*.png",
        args.patch_size,
        device,
        args.patches_per_image
    )

    cv_ratio = 0.2
    n_total = len(dataset)
    n_cv = int(n_total*cv_ratio)
    n_train = n_total - n_cv

    train_dataset, cv_dataset = random_split(
        dataset,
        [n_train, n_cv],
        generator=torch.Generator().manual_seed(25)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    cv_loader = DataLoader(
        cv_dataset,
        batch_size=4,
        shuffle=False, # Don't shuffle validation
        num_workers=0,
        pin_memory=True,
    )

    model = UNet6Residual(channels=3, base=64).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    fit(
        model,
        train_loader,
        cv_loader,
        optimizer,
        loss_fn,
        device,
        epochs=args.epochs,
        output_weights=args.output
    )

def apply_mode(args, device):
    # recreate model
    model = UNet6Residual(channels=3, base=64).to(device)
    state = torch.load(args.weights_in, map_location=device)
    model.load_state_dict(state)

    model.eval()

    noisy = load_image_tensor(args.input, device)
    with torch.no_grad():
        denoised = model(noisy)
    save_image(denoised, args.output)
    print(f"Loaded weights from: {args.weights_in}")
    print(f"Saved denoised image to: {args.output}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, file=sys.stderr)
    
    if args.mode == "train":
        train_mode(args, device)
    elif args.mode == "apply":
        apply_mode(args, device)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
