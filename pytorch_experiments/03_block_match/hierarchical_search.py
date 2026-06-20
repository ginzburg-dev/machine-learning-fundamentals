import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

import structlog

from dataclasses import dataclass

LOGGER = structlog.get_logger(__name__)


_IMAGES = [
    "pytorch_experiments/03_block_match/examples/tennis/tennis.0001.jpg",
    "pytorch_experiments/03_block_match/examples/tennis/tennis.0002.jpg",
    "pytorch_experiments/03_block_match/examples/tennis/tennis.0003.jpg",
]

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_images() -> list[torch.Tensor]:
    """Load images as Tensor."""
    images = []
    for image in _IMAGES:
        t = numpy_to_tensor(plt.imread(image).copy())
        images.append(t)
        
    LOGGER.info("Images loaded sucessfully", count=len(images), device=_DEVICE)
    return images


def numpy_to_tensor(array_: np.ndarray) -> torch.Tensor:
    """Convert NDArray to Tensor."""
    t = torch.from_numpy(array_.copy()).float().to(device=_DEVICE) / 255.0
    alpha = torch.ones_like(t)[..., :1]
    t = torch.cat([t, alpha], dim=-1) 
    t = t.permute(2, 0, 1).unsqueeze(0) 
    return t


def convert_flow2_to_flow3(flow: torch.Tensor) -> torch.Tensor:
    return torch.cat([flow, torch.zeros(1, *flow.size()[1:])], dim=0)


def image_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    return t.permute(1, 2, 0).detach().cpu().numpy()[...,:3]


def mix2tensors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_rgb = source[:, :3, ...]
    target_rgb = target[:, :3, ...]

    source_alpha = source[:, 3:4, ...]
    target_alpha = target[:, 3:4, ...]

    numerator = source_rgb*source_alpha + target_rgb
    weighed_sum = source_alpha + 1

    mixed_rgb = torch.where(weighed_sum > 0, numerator/weighed_sum, torch.zeros_like(source_rgb))

    mixed_alpha = torch.max(source_alpha, target_alpha)

    return torch.cat([mixed_rgb, mixed_alpha], dim=1)


def wrap_tensor(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    b, c, h, w = t.shape
    flow_xy = grid[:2, ...]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=t.device, dtype=torch.float32),
        torch.arange(w, device=t.device, dtype=torch.float32),
        indexing="ij"
    )
    base_grid = torch.stack([grid_x, grid_y], dim=0)
    real_offset = base_grid + flow_xy

    grid_tensor = real_offset.permute(1, 2, 0).unsqueeze(0)
    grid_tensor[..., 0] = 2*grid_tensor[..., 0]/(w - 1) - 1
    grid_tensor[..., 1] = 2*grid_tensor[..., 1]/(h - 1) - 1

    wrapped_rgb = F.grid_sample(
        t[:, :3, ...],
        grid_tensor,
        padding_mode="zeros",
        align_corners=True
    )
    alpha = grid[2:3, ...].unsqueeze(0)

    return torch.cat([wrapped_rgb, alpha], dim=1)


def compute_offset(
        img1: torch.Tensor,
        img2: torch.Tensor,
        search_radius: int,
        window_size: int,
        motion_threshold: float,
        guide: torch.Tensor | None = None,
) -> torch.Tensor:
    """Search offsets."""
    window_size = max(3, window_size | 1)

    if guide is not None:
        img2 = wrap_tensor(img2, guide)

    offsets = [ (dx, dy)    for dy in range(-search_radius, search_radius + 1)
                            for dx in range(-search_radius, search_radius + 1)]
    
    img1_array = []
    img2_array = []
    for dx, dy in offsets:
        shifted = torch.roll(img2, shifts=(dy, dx), dims=(-2, -1))
        
        if dx > 0: shifted[..., :, :dx] = 0
        elif dx < 0: shifted[..., :, dx:] = 0
        if dy > 0: shifted[..., :dy, :] = 0
        elif dy < 0: shifted[..., dy:, :] = 0

        img2_array.append(shifted)
        img1_array.append(img1)

    img1_cat = torch.cat(img1_array, dim=0)
    img2_cat = torch.cat(img2_array, dim=0)

    squared_diff = (img1_cat[:, :3, ...] - img2_cat[:, :3, ...])**2

    windows_area = window_size * window_size
    local_batch_ssd = F.avg_pool2d(
        squared_diff,
        kernel_size=window_size,
        stride=1,
        padding=window_size//2,
    ) * windows_area

    patch_mse = torch.sum(local_batch_ssd, dim=1) / (3 * windows_area)
    mse_min_values, best_indices = torch.min(patch_mse, dim=0)
    offsets_tensor = torch.tensor(offsets, device=img1.device, dtype=torch.float32)
    best_offsets = offsets_tensor[best_indices]

    flow_dx = -best_offsets[..., 0]
    flow_dy = -best_offsets[..., 1]

    motion_mask = torch.where(mse_min_values > motion_threshold, 0.0, 1.0)

    flow = torch.stack([flow_dx, flow_dy, motion_mask], dim=0)

    if guide is not None:
        flow[:2, ...] = flow[:2, ...] + guide[:2, ...]
        flow[2:3, ...] = flow[2:3, ...] * guide[2:3, ...]

    return flow


def resize_tensor_to(t: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    c_h, c_w = t.shape[-2:]
    t_h, t_w = target_hw

    result = F.interpolate(
        t.unsqueeze(0),
        size=(t_h, t_w),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)

    result[0] *= t_w / c_w
    result[1] *= t_h / c_h

    return result


def build_pyramid(t: torch.Tensor, n_levels: int = 3) -> list[torch.Tensor]:
    pyramid = [t]
    for i in range(n_levels):
        t = F.avg_pool2d(t, kernel_size=2)
        pyramid.append(t)
    return pyramid


def wrap_images(
    target_: torch.Tensor,
    source_: torch.Tensor,
    window_size: int = 15,
    search_radius: int  = 10,
    motion_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ Wrap 2 images. """

    img1_levels = build_pyramid(target_)
    img2_levels = build_pyramid(source_)

    reversed_range_ = list(range(len(img1_levels)))[::-1]
    guide = None
    flow: torch.Tensor = torch.zeros(0)
    for i in reversed_range_:
        level_scale = 2 ** i
        current_radius = max(2, search_radius // level_scale)
        current_window = max(5, (window_size // level_scale) | 1) 

        flow = compute_offset(
            img1_levels[i],
            img2_levels[i],
            current_radius,
            current_window,
            motion_threshold,
            guide)
        if i > 0:
            new_h, new_w = img1_levels[i-1].shape[-2:]
            guide = resize_tensor_to(flow, (new_h, new_w))

    return wrap_tensor(source_, flow), flow

def wrap_tennis_1_frame(window_size: int = 7, search_radius: int = 10, threshold: float = 1, debug: bool = False) -> None:
    filepath = Path("/Users/dmitryginzburg/Yandex.Disk.localized/MG_block_match/jpg/")

    frames = sorted([
        file for file in filepath.iterdir()
        if file.suffix.lower() == ".jpg" and 
        "wrapped" not in file.name 
    ])

    for i in range(1, len(frames)):
        LOGGER.info("Processing image", name=frames[i-1].name)

        target_tensor = numpy_to_tensor(plt.imread(frames[i-1]))
        source_tensor = numpy_to_tensor(plt.imread(frames[i]))
        wrapped, flow = wrap_images(target_tensor, source_tensor, window_size, search_radius, threshold)
        name_split = frames[i-1].stem.split(".")
        name = name_split[0] + "_wrapped." + name_split[1] + ".jpg"
        out = frames[i-1].parent / name
        alpha = wrapped[:, 3:4, ...]
        alpha_rgba = torch.cat([alpha, alpha, alpha, alpha], dim=1)

        tensor_mix = mix2tensors(wrapped, target_tensor)

        if debug:
            out_tensor = torch.cat(
                        [
                            tensor_mix,
                            target_tensor,
                            (tensor_mix - target_tensor).abs() * 0.5 + 0.5,
                            alpha_rgba
                        ],
                        dim=3
            )
        else:
            out_tensor = tensor_mix

        plt.imsave(out, image_to_numpy(out_tensor).clip(0, 1)) 



def main() -> None:
    images = load_images()

    img1 = images[0]
    img2 = images[1]

    wrapped, flow = wrap_images(img1, img2, 16, 10)

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))
    ax[0].imshow(image_to_numpy(img1)); ax[0].set_title("img1 (target)")
    ax[1].imshow(image_to_numpy(img2)); ax[1].set_title("img2 (source)")
    ax[2].imshow(image_to_numpy(convert_flow2_to_flow3(flow))); ax[2].set_title("flow")
    ax[3].imshow(image_to_numpy(wrapped)); ax[3].set_title("wrapped")
    ax[4].imshow(image_to_numpy((img1 - wrapped).abs())*0.5 + 0.5); ax[4].set_title("residual")

    plt.show()
    

if __name__ == "__main__":
    #main()
    wrap_tennis_1_frame(11, 24, 0.03, debug=True)
