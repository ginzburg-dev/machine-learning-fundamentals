import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from PIL import Image

# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)   # noisy image
    p.add_argument("--target", required=True)  # clean image
    p.add_argument("--output", required=True)  # denoised output
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


# ---------------------------
# Model: tiny residual UNet-ish
# ---------------------------
class TinyDenoiser(nn.Module):
    """Predicts NOISE, so output = noisy - noise_pred"""

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
        noise_pred = self.decoder(x)  # predicted NOISE
        clean_pred = noisy - noise_pred
        return clean_pred


# ---------------------------
# Utils: load / save / patches
# ---------------------------
def load_image_tensor(path: str, device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()  # 0–1
    x = t(img).unsqueeze(0).to(device)  # [1, C, H, W]
    return x


def save_image(tensor: torch.Tensor, path: str) -> None:
    t = transforms.ToPILImage()
    img_tensor = tensor.detach().cpu().squeeze(0)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    img = t(img_tensor)
    img.save(path)


def random_patch_pair(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    noisy, clean: [1, C, H, W]
    returns patch_noisy, patch_clean: [1, C, ph, pw]
    """
    _, _, H, W = noisy.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image is smaller than patch size {patch_size}")

    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)

    patch_noisy = noisy[:, :, y:y+patch_size, x:x+patch_size]
    patch_clean = clean[:, :, y:y+patch_size, x:x+patch_size]
    return patch_noisy, patch_clean


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, file=sys.stderr)

    # 1) Load full images
    noisy_full = load_image_tensor(args.input, device)
    clean_full = load_image_tensor(args.target, device)

    # 2) Model + optimizer + loss
    model = TinyDenoiser(channels=3, base=64).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()  # L1 often looks nicer for denoising

    # 3) Training on random patches
    total_steps = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()

        #for step in range(args.steps_per_epoch):
            # patch_noisy, patch_clean = random_patch_pair(
            #     noisy_full, clean_full, args.patch_size
            # )
        patch_noisy = noisy_full
        patch_clean = clean_full

        optimizer.zero_grad()
        pred_clean = model(patch_noisy)
        loss = loss_fn(pred_clean, patch_clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_steps += 1

        avg_loss = running_loss / args.steps_per_epoch
        print(f"Epoch {epoch+1}/{args.epochs}, loss = {loss.item():.6f},",
            f"PROGRESS: {int(((epoch+1)/args.epochs)*100)}%")
        sys.stdout.flush()

    # 4) Apply model to full image
    model.eval()
    with torch.no_grad():
        denoised_full = model(noisy_full)

    save_image(denoised_full, args.output)
    print(f"Saved denoised image to: {args.output}")


if __name__ == "__main__":
    main()
