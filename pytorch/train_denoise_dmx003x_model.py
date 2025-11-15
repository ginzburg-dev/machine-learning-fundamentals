import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "apply"], required=True)

    # common
    p.add_argument("--input", required=True, help="noisy image")
    p.add_argument("--output", required=True, help="denoised output")

    # for training
    p.add_argument("--target", help="clean image (for training)")
    p.add_argument("--weights-out", default="denoiser.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    # for apply
    p.add_argument("--weights-in", default="denoiser.pth")

    return p.parse_args()


class TinyDenoiser(nn.Module):
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


def load_image_tensor(path: str, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()  # [0,1]
    x = t(img).unsqueeze(0).to(device)
    return x


def save_image(tensor: torch.Tensor, path: str) -> None:
    t = transforms.ToPILImage()
    img_tensor = tensor.detach().cpu().squeeze(0)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    img = t(img_tensor)
    img.save(path)


def random_patch_pair(noisy, clean, patch_size: int):
    _, _, H, W = noisy.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image smaller than patch size {patch_size}")
    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return (
        noisy[:, :, y:y+patch_size, x:x+patch_size],
        clean[:, :, y:y+patch_size, x:x+patch_size],
    )


def train_mode(args, device):
    if not args.target:
        raise ValueError("--target (clean image) is required in train mode")

    noisy_full = load_image_tensor(args.input, device)
    clean_full = load_image_tensor(args.target, device)

    model = TinyDenoiser(channels=3, base=64).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for _ in range(args.steps_per_epoch):
            patch_noisy, patch_clean = random_patch_pair(
                noisy_full, clean_full, args.patch_size
            )

            optimizer.zero_grad()
            pred_clean = model(patch_noisy)
            loss = loss_fn(pred_clean, patch_clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / args.steps_per_epoch
        print(f"Epoch {epoch+1}/{args.epochs}, loss = {loss.item():.6f},",
              f"PROGRESS: {int(((epoch+1)/args.epochs)*100)}%")

    # save weights
    torch.save(model.state_dict(), args.weights_out)
    print(f"Saved weights to: {args.weights_out}")

    # also denoise the training image once
    model.eval()
    with torch.no_grad():
        denoised_full = model(noisy_full)
    save_image(denoised_full, args.output)
    print(f"Saved denoised training image to: {args.output}")


def apply_mode(args, device):
    # recreate model
    model = TinyDenoiser(channels=3, base=64).to(device)
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
