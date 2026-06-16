import random
import re
from pathlib import Path
from typing import Tuple, Sequence, Any, List, Dict, NamedTuple, Set

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import OpenEXR, Imath
import numpy as np


class SampleInfo(NamedTuple):
    noisy: Path
    clean: Path
    noise_level: str
    subset: str
    aovs_noisy: dict[str, Path]
    aovs_clean: dict[str, Path]


class DenoiserPatchDataset(Dataset):
    def __init__(
            self,
            input_dir: str | Path,
            patch_size: int,
            noise_level: Sequence[str] = ("high", "low", "verylow", "aggressive", "extreme", "mid"),
            subset: Sequence[str] = ("anim"),
            layers: Sequence[str] = ("chars", "env"),
            cache_images: bool = True,
            split_by_patches: bool= True,
            patches_per_image: int = 100,
            n_first_samples: int | None = None,
            n_first_frames: int | None = None
    ) -> None:
        self.input_dir = Path(input_dir)
        self.samples: list[SampleInfo] = []
        for layer in layers:
            self.samples.extend(collect_clean_noisy_samples(
                root=self.input_dir,
                layer=layer,
                noise_levels=("high", "low", "verylow", "aggressive", "extreme", "mid"),
                subsets=("anim"),
                aovs=("rgba"),
                ext=".exr",
                n_first_samples=n_first_samples,
                n_first_frames=n_first_frames
            )
        )
        print(f"[dataset] collected {len(self.samples)} pairs")

        for s in self.samples:
            print("noisy:", s.noisy)
            print("clean:", s.clean)
            print("aovs_clean:", list(s.aovs_clean.keys()))
            print("aovs_noisy:", list(s.aovs_noisy.keys()))

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.split_by_patches = split_by_patches
        self.cache_images = cache_images

        self._noisy_imgs: list[torch.Tensor] = []
        self._clean_imgs: list[torch.Tensor] = []

        if self.split_by_patches and self.patches_per_image <= 0:
            raise ValueError("patches_per_image must be > 0 when split_by_patches=True")

        if self.cache_images:
            print(f"[dataset] Caching {len(self.samples)} image samples...", flush=True)
            for s in self.samples:
                noisy, _ = load_image_tensor(s.noisy)
                clean, _ = load_image_tensor(s.clean)
                if noisy.ndim == 4 and noisy.shape[0] == 1:
                    noisy = noisy.squeeze(0)
                if clean.ndim == 4 and clean.shape[0] == 1:
                    clean = clean.squeeze(0)
                self._noisy_imgs.append(noisy)
                self._clean_imgs.append(clean)

    def __len__(self) -> int:
        if self.split_by_patches:
            return len(self.samples) * self.patches_per_image
        else:
            return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        if self.split_by_patches:
            img_idx = index // self.patches_per_image
        else:
            img_idx = index

        if self.cache_images:
            noisy = self._noisy_imgs[img_idx]
            clean = self._clean_imgs[img_idx]
        else:
            s = self.samples[img_idx]
            noisy, _ = load_image_tensor(s.noisy)
            clean, _ = load_image_tensor(s.clean)

        if noisy.ndim == 4 and noisy.shape[0] == 1:
            noisy = noisy.squeeze(0)
        if clean.ndim == 4 and clean.shape[0] == 1:
            clean = clean.squeeze(0)

        if noisy.shape != clean.shape:
            _, h_n, w_n = noisy.shape
            _, h_c, w_c = clean.shape
            h = min(h_n, h_c)
            w = min(w_n, w_c)
            noisy = noisy[:, :h, :w]
            clean = clean[:, :h, :w]

        if not self.split_by_patches:
            return noisy, clean

        noisy_patch, clean_patch = random_patch_pair(noisy, clean, self.patch_size)
        if self.split_by_patches:
            if random.random() < 0.5:
                noisy_patch = torch.flip(noisy_patch, dims=[-1])
                clean_patch = torch.flip(clean_patch, dims=[-1])
            if random.random() < 0.5:
                noisy_patch = torch.flip(noisy_patch, dims=[-2])
                clean_patch = torch.flip(clean_patch, dims=[-2])
    
        return noisy_patch, clean_patch

def frame_key(path: Path) -> str:
    m = re.search(r"\.(\d+)\.[^.]+$", path.name)
    if m:
        return m.group(1)
    else:
        return path.stem


def collect_clean_noisy_samples(
    root: Path,
    layer: str = "chars",
    noise_levels: Sequence[str] = ("high",),
    subsets: Sequence[str] = ("anim",),
    aovs: Sequence[str] = (),
    ext: str = ".exr",
    n_first_samples: int | None = None,
    n_first_frames: int | None = None
) -> list[SampleInfo]:
    root = Path(root)
    samples: list[SampleInfo] = []

    for shot_dir in sorted(root.iterdir()):
        if not shot_dir.is_dir():
            continue

        rgba_dir = shot_dir / layer / "rgba"
        clean_dir = rgba_dir / "clean"
        if not clean_dir.is_dir():
            continue

        clean_rgba_by_frame: dict[str, Path] = {}
        for clean_rgba in clean_dir.glob(f"*{ext}"):
            fk = frame_key(clean_rgba)
            clean_rgba_by_frame[fk] = clean_rgba

        aov_clean_by_frame: dict[str, dict[str, Path]] = {}
        for aov_name in aovs:
            aov_clean_dir = shot_dir / layer / aov_name / "clean"
            mapping: dict[str, Path] = {}
            if aov_clean_dir.is_dir():
                for p in aov_clean_dir.glob(f"*{ext}"):
                    fk = frame_key(p)
                    mapping[fk] = p
            aov_clean_by_frame[aov_name] = mapping

        for noise_level in noise_levels:
            for subset in subsets:
                noisy_rgba_dir = rgba_dir / "noisy" / noise_level / subset
                if not noisy_rgba_dir.is_dir():
                    continue

                noisy_rgba_by_frame: dict[str, Path] = {}
                for noisy_rgba in noisy_rgba_dir.glob(f"*{ext}"):
                    fk = frame_key(noisy_rgba)
                    noisy_rgba_by_frame[fk] = noisy_rgba

                aov_noisy_by_frame: dict[str, dict[str, Path]] = {}
                for aov_name in aovs:
                    aov_noisy_dir = shot_dir / layer / aov_name / "noisy" / noise_level / subset
                    mapping: dict[str, Path] = {}
                    if aov_noisy_dir.is_dir():
                        for p in aov_noisy_dir.glob(f"*{ext}"):
                            fk = frame_key(p)
                            mapping[fk] = p
                    aov_noisy_by_frame[aov_name] = mapping
                
                seen_frames = 0
                for fk, clean_rgba in clean_rgba_by_frame.items():
                    if n_first_frames is not None and seen_frames >= n_first_frames:
                        break

                    noisy_rgba = noisy_rgba_by_frame.get(fk)
                    if noisy_rgba is None:
                        continue

                    aovs_clean: dict[str, Path] = {}
                    aovs_noisy: dict[str, Path] = {}

                    for aov_name in aovs:
                        clean_aov = aov_clean_by_frame.get(aov_name, {}).get(fk)
                        noisy_aov = aov_noisy_by_frame.get(aov_name, {}).get(fk)
                        if clean_aov is not None:
                            aovs_clean[aov_name] = clean_aov
                        if noisy_aov is not None:
                            aovs_noisy[aov_name] = noisy_aov

                    samples.append(
                        SampleInfo(
                            noisy=noisy_rgba,
                            clean=clean_rgba,
                            noise_level=noise_level,
                            subset=subset,
                            aovs_noisy=aovs_noisy,
                            aovs_clean=aovs_clean,
                        )
                    )

                    if n_first_frames is not None:
                        seen_frames += 1

    if n_first_samples:
        samples = samples[:n_first_samples]
    print(f"Collected {len(samples)} pairs")
    return samples


def build_pairs(input_dir: Path | str, pattern: str = "*.png"):
    input_dir = Path(input_dir)
    noisy_dir = input_dir / "noisy"
    clean_dir = input_dir / "clean"

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


def load_pil_tensor(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()
    x = t(img).unsqueeze(0)
    return x


def save_pil_tensor(tensor: torch.Tensor, path: Path) -> None:
    t = transforms.ToPILImage()
    img_tensor = tensor.detach().cpu().squeeze(0)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    img = t(img_tensor)
    img.save(path)


def load_exr_tensor(path: str | Path) -> tuple[torch.Tensor, dict]:
    path = str(path)
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header().copy()

    dw = header["dataWindow"]
    width  = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for c in ("R", "G", "B"):
        if c not in header["channels"]:
            raise ValueError(f"Channel {c} not found in EXR: {path}")
        ch_str = exr_file.channel(c, pt)
        ch = np.frombuffer(ch_str, dtype=np.float32)
        ch = ch.reshape((height, width))
        channels.append(ch)

    img = np.stack(channels, axis=0)
    tensor = torch.from_numpy(img)
    
    return tensor, header

def save_exr_tensor(t: torch.Tensor, path: str | Path, template_header: dict | None = None):
    path = str(path)

    if t.dim() == 4 and t.shape[0] == 1:
        t = t[0]
    assert t.dim() == 3, f"Expected (C,H,W), got {t.shape}"

    t = t.detach().cpu().float()
    c, h, w = t.shape

    if c < 3:
        raise ValueError(f"Need at least 3 channels (R,G,B), got {c}")

    rgb = t[:3].numpy()

    if template_header is not None:
        dw = template_header["dataWindow"]
        tw = dw.max.x - dw.min.x + 1
        th = dw.max.y - dw.min.y + 1

        if (tw, th) != (w, h):
            print(
                f"[save_exr_tensor] WARNING: tensor ({w}x{h}) "
                f"!= dataWindow ({tw}x{th}), using tensor size."
            )
            width, height = w, h
            dw = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(w - 1, h - 1))
            display_window = dw
        else:
            width, height = tw, th
            display_window = template_header.get("displayWindow", dw)

        header = OpenEXR.Header(width, height)

        for k, v in template_header.items():
            if k in ("channels", "dataWindow", "displayWindow"):
                continue
            header[k] = v

        header["dataWindow"] = dw
        header["displayWindow"] = display_window

    else:
        header = OpenEXR.Header(w, h)
        dw = Imath.Box2i(Imath.V2i(0, 0), Imath.V2i(w - 1, h - 1))
        header["dataWindow"] = dw
        header["displayWindow"] = dw

    float_pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = header.get("channels", {})
    channels["R"] = Imath.Channel(float_pt)
    channels["G"] = Imath.Channel(float_pt)
    channels["B"] = Imath.Channel(float_pt)
    header["channels"] = channels

    out = OpenEXR.OutputFile(path, header)
    out.writePixels({
        "R": rgb[0].astype(np.float32).tobytes(),
        "G": rgb[1].astype(np.float32).tobytes(),
        "B": rgb[2].astype(np.float32).tobytes(),
    })
    out.close()

def load_image_tensor(path: Path) -> Tuple[torch.Tensor, dict | None]:
    if path.suffix.lower() == ".exr":
        tensor, header = load_exr_tensor(path)
        return tensor, header
    else:
        tensor = load_pil_tensor(path)
        return tensor, None


def save_image(t: torch.Tensor, path: Path, header: dict | None = None):
    if path.suffix.lower() == ".exr":
        save_exr_tensor(t, path, header)
    else:
        save_pil_tensor(t, path)


def collect_images(input_path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".exr"}

    if input_path.is_dir():
        files = [
            p
            for p in input_path.rglob("*")
            if p.suffix.lower() in exts
        ]
    else:
        files = [input_path] if input_path.suffix.lower() in exts else []

    return sorted(files)


def random_patch_pair(
    input: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C, H, W = input.shape

    if H < patch_size or W < patch_size:
        raise ValueError(f"Image smaller than patch size {patch_size}")

    y = random.randint(0, H - patch_size)
    x = random.randint(0, W - patch_size)
    return (
        input[..., y:y+patch_size, x:x+patch_size],
        target[..., y:y+patch_size, x:x+patch_size],
    )

