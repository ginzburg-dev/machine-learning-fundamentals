import random
from pathlib import Path
from typing import Tuple, Sequence, Any

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image


class DenoiserPatchDataset(Dataset):
    def __init__(
            self,
            input_dir: str | Path,
            pattern: str,
            patch_size: int,
            device: torch.device,
            patches_per_image: int = 100,
    ) -> None:
        self.input_dir = input_dir
        self.pairs = build_pairs(input_dir, pattern)
        self.device = device
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index) -> Any:
        noisy_path, clean_path = self.pairs[index]

        noisy = load_image_tensor(noisy_path, self.device)
        clean = load_image_tensor(clean_path, self.device)

        if self.patch_size is None:
            return noisy, clean
        
        noisy_patch, clean_path, _ = random_patch_pair(noisy, clean, self.patch_size)

        return noisy_patch, clean_path


def build_pairs(input_dir: str | Path, pattern: str = "*.png"):
    noisy_dir = Path(input_dir) / "noisy"
    clean_dir = Path(input_dir) / "clean"

    pairs: list[tuple[Path, Path]] = []

    for noisy_path in sorted(noisy_dir.glob(pattern)):
        name = noisy_path.name
        clean_path = clean_dir / name
        if clean_path.exists():
            pairs.append((noisy_path, clean_path))
        else:
            print(f"[WARN] No clean file for {noisy_path}")

    if not pairs:
        raise RuntimeError("No pairs found, check folders / pattern")

    return pairs


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


def random_patch_pair(
    input: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    alpha: torch.Tensor | None = None,
    max_alpha_search_tries: int = 50,
    alpha_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    _, _, H, W = input.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image smaller than patch size {patch_size}")

    if alpha is not None:
        alpha = alpha[:,3,4, :, :]
        for _ in range(max_alpha_search_tries):
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)

            alpha_patch = alpha[..., y:y+patch_size, x:x+patch_size]
            if alpha_patch.mean().item() > alpha_threshold:
                return (
                    input[..., y:y+patch_size, x:x+patch_size],
                    target[..., y:y+patch_size, x:x+patch_size],
                    True
                )
        # Return not found
        return (
            input[..., 0, 0],
            target[..., 0, 0],
            False
        )

    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return (
        input[..., y:y+patch_size, x:x+patch_size],
        target[..., y:y+patch_size, x:x+patch_size],
        True
    )

