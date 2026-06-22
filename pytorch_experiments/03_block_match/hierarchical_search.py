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


def output_filepath(input_filename: Path, debug: bool = False) -> Path:
    name_split = input_filename.stem.split(".")
    debug_ = "_debug" if debug else ""
    name = name_split[0] + "_warped" + debug_ + "." + name_split[1] + ".jpg"
    return input_filename.parent / name


def numpy_to_tensor(array_: np.ndarray) -> torch.Tensor:
    """Convert NDArray to Tensor."""
    t = torch.from_numpy(array_.copy()).float().to(device=_DEVICE) / 255.0
    alpha = torch.ones_like(t)[..., :1]
    t = torch.cat([t, alpha], dim=-1) 
    t = t.permute(2, 0, 1).unsqueeze(0) 
    return t


def convert_flow2_to_flow3(flow: torch.Tensor) -> torch.Tensor:
    return torch.cat([flow, torch.zeros(1, *flow.size()[1:])], dim=0)


def flow_to_image(flow: torch.Tensor) -> torch.Tensor:
    return torch.cat([flow, torch.zeros_like(flow)[:1, ...]], dim=0).unsqueeze(0)


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


def prefilter_tensor(t: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, kernel_size | 1)

    h, w = t.size()[-2:]
    t = F.avg_pool2d(
        t,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return t

def compute_offset(
        target: torch.Tensor,
        source: torch.Tensor,
        search_radius: int,
        window_size: int,
        prefilter_kernel_size: int,
        motion_threshold: float,
        guide: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search offsets."""
    LOGGER.info("Start computing offsets")

    target = prefilter_tensor(target, prefilter_kernel_size)
    source = prefilter_tensor(source, prefilter_kernel_size)

    window_size = max(1, window_size)

    if guide is not None:
        source = wrap_tensor(source, guide)

    offsets = [ (dx, dy)    for dy in range(-search_radius, search_radius + 1)
                            for dx in range(-search_radius, search_radius + 1)]

    LOGGER.info("Rlling tensor")

    img1_array = []
    img2_array = []
    for dx, dy in offsets:
        shifted = torch.roll(source, shifts=(dy, dx), dims=(-2, -1))
        
        if dx > 0: shifted[..., :, :dx] = 0
        elif dx < 0: shifted[..., :, dx:] = 0
        if dy > 0: shifted[..., :dy, :] = 0
        elif dy < 0: shifted[..., dy:, :] = 0

        img2_array.append(shifted)
        img1_array.append(target)

    LOGGER.info("Calculating ssd")

    img1_cat = torch.cat(img1_array, dim=0)
    img2_cat = torch.cat(img2_array, dim=0)

    squared_diff = (img1_cat[:, :3, ...] - img2_cat[:, :3, ...])**2

    windows_area = window_size * window_size

    LOGGER.info("Calculating average ssd")

    local_batch_ssd = F.avg_pool2d(
        squared_diff,
        kernel_size=window_size,
        stride=window_size,
        padding=0,
    ) * windows_area

    LOGGER.info("Sum channels")

    ssd_local_mono = torch.sum(local_batch_ssd, dim=1)

    LOGGER.info("Calculate min ssd")

    ssd_min_values, best_indices = torch.min(ssd_local_mono, dim=0)
    offsets_tensor = torch.tensor(offsets, device=target.device, dtype=torch.float32)
    best_offsets = offsets_tensor[best_indices]

    flow_dx = -best_offsets[..., 0]
    flow_dy = -best_offsets[..., 1]

    LOGGER.info("Calculate motion threshold")

    motion_mask = torch.where(ssd_min_values > motion_threshold, 0.0, 1.0)

    flow_block = torch.stack([flow_dx, flow_dy, motion_mask], dim=0)

    h, w = target.shape[-2:]
    flow_image = resize_tensor_to(flow_block, (h, w), adaptive_value_scale=False, mode="nearest")

    if guide is not None:
        flow_image[:2, ...] = flow_image[:2, ...] + guide[:2, ...]
        flow_image[2:3, ...] = flow_image[2:3, ...] * guide[2:3, ...]


    ds_h, ds_w = flow_block.shape[-2:]
    total_flow_block = resize_tensor_to(
        flow_image,
        (ds_h, ds_w),
        adaptive_value_scale=False,
        mode="nearest",
    )

    LOGGER.info("Motion flow calcualted succesfully")

    return flow_image, total_flow_block


def resize_tensor_to(
    t: torch.Tensor,
    target_hw: tuple[int, int],
    adaptive_value_scale: bool,
    mode: str = "nearest"
) -> torch.Tensor:
    """ Resize tensor to different dimention. """
    c_h, c_w = t.shape[-2:]
    t_h, t_w = target_hw

    if mode == "nearest":
        result = F.interpolate(
            t.unsqueeze(0),
            size=(t_h, t_w),
            mode=mode,
        ).squeeze(0)
    else:
        result = F.interpolate(
            t.unsqueeze(0),
            size=(t_h, t_w),
            mode=mode,
            align_corners=True
        ).squeeze(0)

    if adaptive_value_scale:
        result[0] *= t_w / c_w
        result[1] *= t_h / c_h

    return result


def build_pyramid(t: torch.Tensor, n_levels: int = 3) -> list[tuple[torch.Tensor, int]]:
    scale = 1
    pyramid = [(t, scale)]
    for i in range(n_levels):
        t = F.avg_pool2d(t, kernel_size=2)
        scale *= 2
        pyramid.append((t, scale))
    return pyramid


def wrap_images(
    target_: torch.Tensor,
    source_: torch.Tensor,
    window_size: int = 15,
    search_radius: int  = 10,
    prefilter_kernel_size: int = 1,
    motion_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ Wrap 2 images. """

    img1_levels = build_pyramid(target_, 5)
    img2_levels = build_pyramid(source_, 5)

    reversed_range_ = list(range(len(img1_levels)))[::-1]
    guide = None
    flow: torch.Tensor = torch.zeros(0)
    for i in reversed_range_:
        level_scale = img1_levels[i][1]
        current_radius = max(1, search_radius)
        #current_radius = max(1, search_radius)
        #current_window = max(1, (window_size // level_scale) | 1) 
        #current_radius = max(1, search_radius)
        current_window = max(8, round(window_size / level_scale))

        flow, flow_block = compute_offset(
            img1_levels[i][0],
            img2_levels[i][0],
            current_radius,
            current_window,
            prefilter_kernel_size,
            motion_threshold,
            guide)

        if i > 0:
            new_h, new_w = img1_levels[i-1][0].shape[-2:]
            guide = resize_tensor_to(
                t=flow, 
                target_hw=(new_h, new_w),
                adaptive_value_scale=True,
                mode="nearest"
            )

    #flow = prefilter_tensor(flow, 3) # OPTIONAL prefilter flow tensor

    assert flow.shape[-2:] == source_.shape[-2:]

    return wrap_tensor(source_, flow), flow


def warp_int_pyramid_frames(
        input_sequence_folder: str,
        window_size: int = 7,
        search_radius: int = 10,
        prefilter_kernel: int = 4,
        threshold: float = 1,
        debug: bool = False
) -> None:
    filepath = Path(input_sequence_folder)

    frames = sorted([
        file for file in filepath.iterdir()
        if file.suffix.lower() == ".jpg" and 
        "warped" not in file.name 
    ])

    for i in range(1, len(frames)):
        LOGGER.info("Processing image", name=frames[i-1].name)

        target_tensor = numpy_to_tensor(plt.imread(frames[i-1]))
        source_tensor = numpy_to_tensor(plt.imread(frames[i]))

        warped, flow = wrap_images(target_tensor, source_tensor, window_size, search_radius, prefilter_kernel, threshold)

        alpha = warped[:, 3:4, ...]
        alpha_rgba = torch.cat([alpha, alpha, alpha, alpha], dim=1)

        tensor_mix = mix2tensors(warped, target_tensor)

        if debug:
            debug_tensor = torch.cat(
                        [
                            tensor_mix,
                            target_tensor,
                            flow_to_image(flow),
                            (tensor_mix - target_tensor).abs() * 0.5 + 0.5,
                            alpha_rgba
                        ],
                        dim=3
            )
            plt.imsave(
                output_filepath(frames[i-1], debug=True),
                image_to_numpy(debug_tensor).clip(0, 1),
                pil_kwargs={
                    "quality": 100,
                    "subsampling": 0,
                },
            )

        plt.imsave(
            output_filepath(frames[i-1]),
            image_to_numpy(tensor_mix).clip(0, 1),
            pil_kwargs={
                "quality": 100,
                "subsampling": 0,
            },
        )


def main() -> None:
    images = load_images()

    img1 = images[0]
    img2 = images[1]

    flow = compute_offset(img1, img2, 16, 10)

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))

    ax[0].imshow(image_to_numpy(img1)); ax[0].set_title("img1 (target)")
    ax[1].imshow(image_to_numpy(img2)); ax[1].set_title("img2 (source)")
    ax[2].imshow(image_to_numpy(convert_flow2_to_flow3(flow))); ax[2].set_title("flow")
    ax[3].imshow(image_to_numpy(wrapped)); ax[3].set_title("wrapped")
    ax[4].imshow(image_to_numpy((img1 - wrapped).abs())*0.5 + 0.5); ax[4].set_title("residual")

    plt.show()


def prefilter_test() -> None:
    images = load_images()

    source = images[0]

    prefiltered = prefilter_tensor(source, 7)

    print(source.shape, prefiltered.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_to_numpy(source))
    ax[1].imshow(image_to_numpy(prefiltered))
    plt.show()


if __name__ == "__main__":
    input_folder = "/Users/dmitryginzburg/Yandex.Disk.localized/MG_block_match/jpg1/"
    warp_int_pyramid_frames(
        input_sequence_folder=input_folder,
        window_size=32,
        search_radius=3,
        prefilter_kernel=3,
        threshold=10000,
        debug=True)

    #prefilter_test()
    #wrap_16_blocks(16, 5, 100, debug=True)
