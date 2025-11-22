import argparse
import random
import sys
import glob
import shutil
from typing import Tuple

import math
import torch.nn.functional as F


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from ml_denoiser.training.models import UNet6Residual, UNetResidual, get_model
from ml_denoiser.training.losses import get_loss
from ml_denoiser.training.training import fit, evaluate
from ml_denoiser.training.utils import save_training_parameters, get_frame_number, get_clean_basename
from ml_denoiser.training.dataset import DenoiserPatchDataset, load_image_tensor, save_image, collect_images

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "apply", "evaluate", "apply_epoch_sequence"], required=True)

    # common
    p.add_argument("--input", required=True, help="input (DIR - for train, FILE - for apply)")
    p.add_argument("--output", help="denoised output")

    # for training
    #p.add_argument("--target", help="clean image (for training)")
    p.add_argument("--model", default="UNetResidual")
    p.add_argument("--loss", default="L1Loss")
    p.add_argument("--model-in-channels", type=int, default=3)
    p.add_argument("--model-out-channels", type=int, default=64)
    p.add_argument("--weights-out", default="denoiser.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patches-per-image", type=int, default=100)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--n-first-samples", type=int, default=None)
    p.add_argument("--n-first-frames", type=int, default=None)
    p.add_argument("--save-checkpoint-every", type=int, default=10)
    p.add_argument("--print-every-n-step", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tensorboard-output", default="tensorboard_logs")

    # for apply
    p.add_argument("--weights-in", default="denoiser.pt")

    return p.parse_args()


def train_mode(args, device):
    #if not args.target:
    #    raise ValueError("--target (clean image) is required in train mode")

    dataset = DenoiserPatchDataset(
        args.input,
        cache_images=True,
        patch_size=args.patch_size,
        split_by_patches=True,
        patches_per_image=args.patches_per_image,
        n_first_samples=args.n_first_samples,
        n_first_frames=args.n_first_frames
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    cv_loader = DataLoader(
        cv_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Don't shuffle validation
        num_workers=0,
        pin_memory=True,
    )

    model = get_model(args.model, args.model_in_channels, args.model_out_channels).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = get_loss(args.loss)

    param_path = Path(args.output) / "training_parameters.json"

    save_training_parameters(
         path=param_path,
         model_name="UNetResidual",
         in_channels=3,
         out_channels=64,
         epochs=args.epochs,
         batch_size=args.batch_size,
         patches_per_image=args.patches_per_image,
         patch_size=args.patch_size,
         n_first_samples=args.n_first_samples,
         n_first_frames=args.n_first_frames,
         lr=args.lr,
    )
    print(f"Training parameters saved to {param_path}")

    fit(
        model=model,
        train_loader=train_loader,
        cv_loader=cv_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_checkpoint_every=args.save_checkpoint_every,
        epochs=args.epochs,
        output_weights=args.weights_out,
        print_every_n_steps=args.print_every_n_step,
        tensorboard_output=args.tensorboard_output,
    )

def apply_model_to_tensor(model, noisy: torch.Tensor, device: torch.device) -> torch.Tensor:
        model.eval()
        noisy = noisy.to(device)
        print(noisy.shape)

        factor = 16  # 2^num_downsamples
        if noisy.dim() == 3:
            noisy = noisy.unsqueeze(0)  # -> (1, C, H, W)

        _, _, h, w = noisy.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        if pad_h != 0 or pad_w != 0:
            noisy_padded = F.pad(
                noisy,
                (0, pad_w, 0, pad_h),
                mode="reflect",
            )
        else:
            noisy_padded = noisy

        with torch.no_grad():
            denoised_padded = model(noisy_padded)

        return denoised_padded[..., :h, :w]


def apply_mode(args, device):
    # recreate model
    model = get_model(args.model, args.model_in_channels, args.model_out_channels).to(device)
    state = torch.load(args.weights_in, map_location=device, weights_only=True)
    if Path(args.weights_in).suffix.lower() == ".json":
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    print(f"Loaded weights from: {args.weights_in}")

    input = Path(args.input)
    if input.is_dir():
        img = collect_images(input)
        print(f"Collected {len(img)} images from {input}")
    else:
        img = [input]
        print(f"Single input image: {input}")

    output = Path(args.output)
    if output.is_dir():
        output_path = output
    else:
        output_path = output.parent

    output_path.mkdir(parents=True, exist_ok=True)

    for n, noisy_path in enumerate(img):
        noisy, header = load_image_tensor(noisy_path)
        denoised = apply_model_to_tensor(model, noisy, device)
        frame = get_frame_number(noisy_path)        # e.g. "0001"
        basename = get_clean_basename(noisy_path)   # e.g. "name"
        if frame is not None:
            output_file = output_path / f"{basename}_denoised.{frame}{noisy_path.suffix}"
        else:
            output_file = output_path / f"{basename}_denoised.{noisy_path.suffix}"
        save_image(denoised, output_file, header)
        print(f"Saved denoised image to: {output_file}")

        if n % 1 == 0:
            progress = int(n /len(img) * 100)
            print(f"PROGRESS: {progress}%")
            sys.stdout.flush()


def apply_epoch_sequence_mode(args, device):
    """Apply model weights from a sequence of epochs to input images.
    in_weights should contain a glob pattern with epoch number placeholder, e.g. denoiser_checkpoint_*.pt.
    """
    model = get_model(args.model, args.model_in_channels, args.model_out_channels).to(device)
    weights_paths = sorted(glob.glob(str(Path(args.weights_in).parent / "*checkpoint*.json")))
    print(f"Found {len(weights_paths)} weights files matching pattern: {args.weights_in}*checkpoint*.json")

    for n, weights_path in enumerate(weights_paths):
        if "checkpoint" not in weights_path:
            raise ValueError("Weights path should contain 'checkpoint' to indicate epoch checkpoints.")

        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
        print(f"Applying weights from: {weights_path}")
        input = Path(args.input)
        if input.is_dir():
            img = collect_images(input)
            print(f"Collected {len(img)} images from {input}")
        else:
            img = [input]
            print(f"Single input image: {input}")

        print(f"Image list: {img}")

        output = Path(args.output)
        if output.is_dir:
            output_path = output
        else:
            output_path = output.parent

        output_path.mkdir(parents=True, exist_ok=True)

        for n, noisy_path in enumerate(img):
            noisy, header = load_image_tensor(noisy_path)
            denoised = apply_model_to_tensor(model, noisy, device)
            frame = get_frame_number(noisy_path)        # e.g. "0001"
            basename = get_clean_basename(noisy_path)   # e.g. "name"
            if frame is not None:
                output_file = output_path / f"{basename}_{frame}_denoised.{int(state["epoch"]):04d}{noisy_path.suffix}"
            else:
                output_file = output_path / f"{basename}_denoised.{frame}{noisy_path.suffix}"
            
            save_image(denoised, output_file, header)
            print(f"Saved denoised image to: {output_file}")

        if n % 1 == 0:
            progress = int(n /len(weights_paths) * 100)
            print(f"PROGRESS: {progress}%")
            sys.stdout.flush()

def evaluate_mode(args, device):
    # recreate model
    model = get_model(args.model, args.model_in_channels, args.model_out_channels).to(device)
    state = torch.load(args.weights_in, map_location=device, weights_only=True)
    model.load_state_dict(state)
    
    dataset = DenoiserPatchDataset(
        input_dir=args.input,
        cache_images=True,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        n_first_frames=args.n_first_frames
    )
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    loss_fn = get_loss(args.loss)

    evaluate(model=model, dataloader=data_loader, loss_fn=loss_fn, device=device)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, file=sys.stderr)
    
    if args.mode == "train":
        train_mode(args, device)
    elif args.mode == "apply":
        apply_mode(args, device)
    elif args.mode == "evaluate":
        evaluate_mode(args, device)
    elif args.mode == "apply_epoch_sequence":
        apply_epoch_sequence_mode(args, device)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
