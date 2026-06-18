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
    "pytorch_experiments/03_block_match/examples/ball.0001.jpg",
    "pytorch_experiments/03_block_match/examples/ball.0002.jpg",
    "pytorch_experiments/03_block_match/examples/ball.0003.jpg",
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
    return t.permute(1, 2, 0).detach().cpu().numpy()


def image_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    return t.permute(1, 2, 0).detach().cpu().numpy() / 255.0


def search_ssd(
        img1: torch.Tensor,
        img2: torch.Tensor,
        search_radius: int,
        window_size: int
) -> torch.Tensor:
    """Search offsets"""
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
    flow_dz = torch.zeros_like(flow_dy)

    flow = torch.stack([flow_dx, flow_dy, flow_dz], dim=0)
    LOGGER.info("flow calculated", shape=flow.shape)
    return flow


def main() -> None:
    images = load_images()
    img1 = F.avg_pool2d(images[0], kernel_size=8)
    img2 = F.avg_pool2d(images[1], kernel_size=8)
    flow = search_ssd(img1, img2, 40, 3)
    plt.imshow(map_to_numpy(flow))
    plt.show()

    t = images[0]
    x = F.avg_pool2d(t, kernel_size=2)
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.avg_pool2d(x, kernel_size=2)
    x = x.roll(shifts=(2, 2), dims=(1, 2))
    x[..., :2, :] = 0
    x[..., :, :2] = 0
    img = x.permute(1, 2, 0).numpy() / 255.0
    

if __name__ == "__main__":
    main()
