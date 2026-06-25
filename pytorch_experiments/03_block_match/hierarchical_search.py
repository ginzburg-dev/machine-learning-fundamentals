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

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def warp_tensor(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
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
        block_size: int,
        prefilter_kernel_size: int,
        motion_threshold: float,
        guide: torch.Tensor | None = None,
) -> torch.Tensor:
    """Search offsets."""
    LOGGER.info("Start computing offsets")

    target = prefilter_tensor(target, prefilter_kernel_size)
    source = prefilter_tensor(source, prefilter_kernel_size)

    current_block_size = max(1, block_size)

    if guide is not None:
        source = warp_tensor(source, guide)

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

    windows_area = current_block_size * current_block_size

    LOGGER.info("Calculating average ssd")

    local_batch_ssd = F.avg_pool2d(
        squared_diff,
        kernel_size=current_block_size,
        stride=current_block_size,
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

    block_mse_min = ssd_min_values / (current_block_size * current_block_size * 3)
    motion_mask = torch.where(
        block_mse_min < motion_threshold,
        1.0,
        0.0,
    )

    flow_block = torch.stack([flow_dx, flow_dy, motion_mask], dim=0)

    LOGGER.info("Motion flow calcualted succesfully")

    return flow_block


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


def compute_pyramid_motion_estimation(
    target_: torch.Tensor,
    source_: torch.Tensor,
    block_size: int,
    search_radius: int,
    n_levels: int,
    prefilter_kernel_size: int,
    motion_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    img1_levels = build_pyramid(target_, n_levels)
    img2_levels = build_pyramid(source_, n_levels)

    reversed_range_ = list(range(len(img1_levels)))[::-1]
    guide: torch.Tensor | None = None
    final_flow_block: torch.Tensor | None = None

    for i in reversed_range_:
        level_scale = img1_levels[i][1]

        current_h, current_w = img1_levels[i][0].shape[-2:]

        current_radius = max(1, search_radius)

        current_block_size = max(1, round(block_size / level_scale))

        flow_block_residual = compute_offset(
            img1_levels[i][0],
            img2_levels[i][0],
            current_radius,
            current_block_size,
            prefilter_kernel_size,
            motion_threshold,
            guide
        )

        flow_residual_dense = resize_tensor_to(
            flow_block_residual,
            (current_h, current_w),
            adaptive_value_scale=False,
            mode="nearest"
        )
        
        if guide is None:
            guide = flow_residual_dense
        else:
            guide[:2, ...] = guide[:2, ...] + flow_residual_dense[:2, ...]
            guide[2:3, ...] = guide[2:3, ...] * flow_residual_dense[2:3, ...]

        if i > 0:
            next_h, next_w = img1_levels[i - 1][0].shape[-2:]
            guide = resize_tensor_to(
                guide,
                (next_h, next_w),
                adaptive_value_scale=True,
                mode="bilinear",
            )
        else:
            flow_xy_block = F.avg_pool2d(
                guide[:2, ...].unsqueeze(0),
                kernel_size=block_size,
                stride=block_size,
            ).squeeze(0)

            mask_block = F.avg_pool2d(
                guide[2:3, ...].unsqueeze(0),
                kernel_size=block_size,
                stride=block_size,
            ).squeeze(0)

            mask_block = (mask_block > 0.5).float()

            final_flow_block = torch.cat([flow_xy_block, mask_block], dim=0)

    assert final_flow_block is not None

    return final_flow_block


def compose_int_pyramid_frames(
        input_sequence_folder: str,
        n_frames: int,
        block_size: int,
        overlap: int,
        search_radius: int,
        n_levels: int,
        prefilter_kernel: int,
        threshold: float,
        debug: bool,
        device: torch.device,
) -> None:
    if not n_frames % 2:
        LOGGER.error("n_frames should be odd. Skipping.")
        return

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

        flow_block = compute_pyramid_motion_estimation(
            target_tensor,
            source_tensor,
            block_size,
            search_radius,
            n_levels,
            prefilter_kernel,
            threshold,
            device,
        )

        composed = compose_patches(source_tensor, flow_block, block_size, overlap, device)

        alpha = composed[:, 3:4, ...]
        alpha_rgba = torch.cat([alpha, alpha, alpha, alpha], dim=1)

        tensor_mix = mix2tensors(composed, target_tensor)

        if debug:
            th, tw = target_tensor.shape[-2:]
            flow_block_vis = resize_tensor_to(flow_block, (th, tw), adaptive_value_scale=False)
            debug_tensor = torch.cat(
                        [
                            tensor_mix,
                            target_tensor,
                            flow_to_image(flow_block_vis),
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


def make_feather_mask(
    patch_size: int,
    overlap: int,
    device: torch.device,
    clip_left: int = 0,
    clip_right: int = 0,
    clip_top: int = 0,
    clip_bottom: int = 0,
) -> torch.Tensor:

    valid_width = patch_size - clip_left - clip_right
    valid_height = patch_size - clip_top - clip_bottom

    fade_in = torch.linspace(0.0, 1.0, overlap, device=device)
    fade_out = torch.linspace(1.0, 0.0, overlap, device=device)

    if patch_size <= 0:
        return torch.zeros((patch_size, patch_size), device=device, dtype=torch.float32)

    x = torch.ones(patch_size, device=device, dtype=torch.float32)
    y = torch.ones(patch_size, device=device, dtype=torch.float32)

    x[:clip_left] = 0.0
    x[patch_size-clip_right:] = 0.0
    y[:clip_top] = 0.0
    y[patch_size-clip_bottom:] = 0.0

    if overlap <= 0:
        return x.unsqueeze(0) * y.unsqueeze(1)

    w_overlap = min(valid_width, overlap)
    h_overlap = min(valid_height, overlap)

    x[clip_left:clip_left + w_overlap] *= fade_in[:w_overlap]
    y[clip_top:clip_top + h_overlap] *= fade_in[:h_overlap]

    x[-w_overlap - clip_right:patch_size-clip_right] *= fade_out[-w_overlap:]
    y[-h_overlap - clip_bottom:patch_size-clip_bottom] *= fade_out[-h_overlap:]

    return x.unsqueeze(0) * y.unsqueeze(1)


def compose_patches(
    source: torch.Tensor,
    flow_block_motion: torch.Tensor,
    block_size: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    h, w = source.shape[-2:]
    bh, bw = flow_block_motion.shape[-2:]
    
    patch_size = block_size + 2 * overlap

    source_rgb = source[:, :3, ...]
    source_alpha = source[:, 3:4, ...]

    accum_buffer = torch.zeros_like(source_rgb, device=device)
    weight_sum = torch.zeros((1, 1, h, w), device=device)
    alpha = torch.zeros((1, 1, h, w), device=device)

    for by in range(bh):
        for bx in range(bw):
            dx = round(flow_block_motion[0, by, bx].item())
            dy = round(flow_block_motion[1, by, bx].item())

            dest_y0_raw = by * block_size - overlap
            dest_x0_raw = bx * block_size - overlap
            dest_y1_raw = by * block_size + block_size + overlap
            dest_x1_raw = bx * block_size + block_size + overlap

            dest_y0 = max(0, dest_y0_raw)
            dest_x0 = max(0, dest_x0_raw)
            dest_y1 = min(h, dest_y1_raw)
            dest_x1 = min(w, dest_x1_raw)

            src_y0 = dest_y0 + dy
            src_x0 = dest_x0 + dx
            src_y1 = dest_y1 + dy
            src_x1 = dest_x1 + dx

            clip_left = 0
            clip_right = 0
            clip_top = 0
            clip_bottom = 0

            if src_y0 < 0:
                shift =  -src_y0
                clip_top = shift
                src_y0 += shift
                dest_y0 += shift
            if src_x0 < 0:
                shift =  -src_x0
                clip_left = shift
                src_x0 += shift
                dest_x0 += shift
            if src_y1 > h:
                shift = src_y1 - h
                clip_bottom = shift
                src_y1 -= shift
                dest_y1 -= shift
            if src_x1 > w:
                shift = src_x1 - w
                clip_right = shift
                src_x1 -= shift
                dest_x1 -= shift
            
            if dest_x1 <= dest_x0 or dest_y1 <= dest_y0:
                continue
            
            w0 = dest_x0 - dest_x0_raw
            h0 = dest_y0 - dest_y0_raw
            w1 = w0 + (dest_x1 - dest_x0)
            h1 = h0 + (dest_y1 - dest_y0)

            patch_rgb = source_rgb[..., src_y0:src_y1, src_x0:src_x1]
            patch_alpha = source_alpha[..., src_y0:src_y1, src_x0:src_x1]

            base_weights = make_feather_mask(
                patch_size=patch_size,
                overlap=overlap,
                device=device,
                clip_left=clip_left,
                clip_right=clip_right,
                clip_top=clip_top,
                clip_bottom=clip_bottom,
            )

            weights = base_weights[..., h0:h1, w0:w1]
            weights = weights.view(1, 1, dest_y1 - dest_y0, dest_x1 - dest_x0)

            weighted_alpha = patch_alpha * weights
            
            accum_buffer[..., dest_y0:dest_y1, dest_x0:dest_x1] += \
                patch_rgb * weighted_alpha
            weight_sum[..., dest_y0:dest_y1, dest_x0:dest_x1] += weighted_alpha

    composite_rgb = torch.where(
        weight_sum > 1e-6,
        accum_buffer/weight_sum.clamp_min(1e-6),
        torch.zeros_like(accum_buffer)
    )

    alpha = weight_sum.clamp(0.0, 1.0)

    return torch.cat([composite_rgb, alpha], dim=1)


def prefilter_test() -> None:
    images = load_images()

    source = images[0]

    prefiltered = prefilter_tensor(source, 7)

    print(source.shape, prefiltered.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_to_numpy(source))
    ax[1].imshow(image_to_numpy(prefiltered))
    plt.show()


def test_feather_mask() -> None:
    mask = make_feather_mask(
                32,
                8,
                _DEVICE,
                0, 0, 0, 0,
            ).unsqueeze(0)
    print(mask.shape)
    plt.imshow(image_to_numpy(mask))
    plt.show()

def test_frames() -> None:
    frames1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    n_frames = 27
    length = len(frames1)
    temporal_radius = n_frames // 2

    (length - n_frames) // 2

    tail_start = []
    tail_end = []
    for i in range(1, temporal_radius + 1):
        index = min(i, length-1)
        tail_start.append(frames1[index])
    tail_end = [ length - i - 1 for i in tail_start]
    result = tail_start[::-1]
    result.extend(frames1)
    result.extend(tail_end)
    print(result)



if __name__ == "__main__":
    input_folder = "/Users/dmitryginzburg/Yandex.Disk.localized/MG_block_match/jpg1/"
    test_frames()
    # compose_int_pyramid_frames(
    #     input_sequence_folder=input_folder,
    #     n_frames=3,
    #     block_size=32,
    #     overlap=8,
    #     search_radius= 3,
    #     n_levels=3,
    #     prefilter_kernel=3,
    #     threshold=0.005,
    #     debug=True,
    #     device=_DEVICE,
    # )

    #test_feather_mask()
    #prefilter_test()
    #wrap_16_blocks(16, 5, 100, debug=True)
