import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

import structlog

from dataclasses import dataclass

LOGGER = structlog.get_logger(__name__)


@dataclass
class Point2d:
    x: int
    y: int

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
        t = torch.from_numpy(plt.imread(image).copy()).float().to(device=_DEVICE)
        t = t.permute(2, 0, 1).unsqueeze(0)
        images.append(t)
    LOGGER.info("loaded images", count=len(images), device=_DEVICE)
    return images


def map_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    print(t.shape)
    return t.permute(1, 2, 0).detach().cpu().numpy()


def image_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    return t.permute(1, 2, 0).detach().cpu().numpy() / 255.0


def wrap(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    b, c, h, w = t.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=_DEVICE, dtype=torch.float32),
        torch.arange(w, device=_DEVICE, dtype=torch.float32),
        indexing="ij"
    )
    base_grid = torch.stack([grid_x, grid_y], dim=0)
    print(base_grid.shape, grid.shape)
    real_offset = base_grid - grid
    grid_tensor = real_offset.permute(1, 2, 0).unsqueeze(0)
    grid_tensor[..., 0] = 2*grid_tensor[..., 0]/(w - 1) - 1
    grid_tensor[..., 1] = 2*grid_tensor[..., 1]/(h - 1) - 1

    return F.grid_sample(t, grid_tensor, align_corners=False)


def compute_offset(
        img1: torch.Tensor,
        img2: torch.Tensor,
        search_radius: int,
        window_size: int,
        guide: torch.Tensor | None = None,
) -> torch.Tensor:
    """Search offsets."""
    if window_size % 2 == 0:
        raise ValueError("winsow_size should be odd.")

    if guide is not None:
        img2 = wrap(img2, guide)

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

    squared_diff = (img1_cat - img2_cat)**2

    windows_area = window_size * window_size
    local_batch_ssd = F.avg_pool2d(
        squared_diff,
        kernel_size=window_size,
        stride=1,
        padding=window_size//2,
    ) * windows_area

    mono_local_batch_ssd = torch.sum(local_batch_ssd, dim=1)
    ssd_min_values, best_indices = torch.min(mono_local_batch_ssd, dim=0)
    offsets_tensor = torch.tensor(offsets, device=img1.device, dtype=torch.float32)
    best_offsets = offsets_tensor[best_indices]

    flow_dx = best_offsets[..., 0]
    flow_dy = best_offsets[..., 1]

    flow = torch.stack([flow_dx, flow_dy], dim=0)

    if guide is not None:
        flow = flow + guide

    return flow


def build_pyramid(t: torch.Tensor, n_levels: int = 3) -> list[tuple[torch.Tensor, int]]:
    pyramid = [(t, 0)]
    for i in range(n_levels):
        t = F.avg_pool2d(t, kernel_size=2)
        pyramid.append((t, i))
    return pyramid


def main() -> None:
    window_size = 15
    search_radius = 2

    images = load_images()

    img1_levels = build_pyramid(images[0])
    img2_levels = build_pyramid(images[2])

    reversed_range_ = list(range(len(img1_levels)))[::-1]
    guide = None
    flow: torch.Tensor = torch.zeros(0)
    for i in reversed_range_:
        flow = compute_offset(
            img1_levels[i][0],
            img2_levels[i][0],
            search_radius,
            window_size,
            guide)
        if i > 0:
            guide = F.interpolate(
                flow.unsqueeze(0),
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) * 2

    wrapped = wrap(images[1], flow)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(image_to_numpy(images[0])); ax[0].set_title("img1 (target)")
    ax[1].imshow(image_to_numpy(images[1])); ax[1].set_title("img2 (source)")
    ax[2].imshow(image_to_numpy(wrapped)); ax[2].set_title("wrapped")
    ax[3].imshow(image_to_numpy((images[0] - wrapped).abs())); ax[3].set_title("residual")

    plt.show()
    

if __name__ == "__main__":
    main()
