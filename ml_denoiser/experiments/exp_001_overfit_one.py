import os
import sys
import af # pyright: ignore[reportMissingImports]
import argparse

from pathlib import Path

from ml_denoiser.config import WORKING_DIR, AF_WRAPPER_PATH, TRAINER_APP_PATH, TGB_DATASET_DIR, ML_DENOISER_DIR
from ml_denoiser.utils.submit_af_job import Command, CommandBlock, JobConfig, submit_job
from ml_denoiser.utils.tensorboard_run import launch_tensorboard

EXPERIMENT_NAME = Path(__file__).stem
WEIGHTS_OUT_NAME = EXPERIMENT_NAME + "_weights.pt"

OUTPUT_DIR = ML_DENOISER_DIR / "output" / "experiments" / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_OUT_PATH = OUTPUT_DIR / "weights"
WEIGHTS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

WEIGHTS_FILE_OUT_PATH = OUTPUT_DIR / "weights" / WEIGHTS_OUT_NAME
WEIGHTS_FILE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TENSORBOARD_LOG_DIR = OUTPUT_DIR / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_DATASET_DIR = ML_DENOISER_DIR / "datasets" / "TGB" / "TGB001" / "validation" / "noisy"

OUTPUT_IMAGES_DIR = ML_DENOISER_DIR / "output" / "experiments" / EXPERIMENT_NAME / "images"
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPATIAL_DIR = OUTPUT_IMAGES_DIR / "spatial_denoised_validation"
OUTPUT_SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_EPOCH_SEQUENCE_DIR = OUTPUT_IMAGES_DIR / "epoch_sequence"
OUTPUT_EPOCH_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TEMPORAL_SEQUENCE_DIR = OUTPUT_IMAGES_DIR / "temporal_sequence"
OUTPUT_TEMPORAL_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

def submit():
    model = "UNetResidual"
    loss = "L1Loss"
    model_in_channels = 3
    model_out_channels = 64
    weights_in = WEIGHTS_OUT_PATH
    weights_out = WEIGHTS_OUT_PATH
    epochs = 240
    batch_size = 8
    patches_per_image = 200
    patch_size = 128
    n_first_samples = 200
    n_first_frames = 3
    save_checkpoint_every = 10
    print_every_n_steps = 1
    lr = 1e-4
    

    job_name = f"torch-{EXPERIMENT_NAME}-job"
    train_command = [
        AF_WRAPPER_PATH,
        "python",
        TRAINER_APP_PATH,
        "--mode train",
        f"--model {model}",
        f"--model-in-channels {model_in_channels}",
        f"--model-out-channels {model_out_channels}",
        f"--loss {loss}",
        f"--input {TGB_DATASET_DIR}",
        f"--output {OUTPUT_DIR}",
        f"--weights-out {weights_out}",
        f"--weights-in {weights_in}",
        f"--epochs {epochs}",
        f"--batch-size {batch_size}",
        f"--patches-per-image {patches_per_image}",
        f"--patch-size {patch_size}",
        f"--n-first-samples {n_first_samples}",
        f"--n-first-frames {n_first_frames}",
        f"--save-checkpoint-every {save_checkpoint_every}",
        f"--print-every-n-step {print_every_n_steps}",
        f"--lr {lr}",
        f"--tensorboard-output {TENSORBOARD_LOG_DIR}"
    ]
    train_command = " ".join(map(str, train_command))

    spatial_validation_command= [
        AF_WRAPPER_PATH,
        "python",
        TRAINER_APP_PATH,
        "--mode apply",
        f"--input {VALIDATION_DATASET_DIR}",
        f"--output {OUTPUT_SPATIAL_DIR}",
        f"--weights-in {weights_in}",
    ]
    spatial_validation_command = " ".join(map(str, spatial_validation_command))

    seqence_over_epoch_command = [
        AF_WRAPPER_PATH,
        "python",
        TRAINER_APP_PATH,
        "--mode apply_epoch_sequence",
        f"--input {VALIDATION_DATASET_DIR}",
        f"--output {OUTPUT_EPOCH_SEQUENCE_DIR}",
        f"--weights-in {weights_in}",
    ]
    seqence_over_epoch_command = " ".join(map(str, seqence_over_epoch_command))


    char_human_closeup_temporal_sequence = TGB_DATASET_DIR / "TGB1004140" / "chars" / "rgba" / "noisy"
    out = OUTPUT_TEMPORAL_SEQUENCE_DIR / 'TGB1004140_char_human_closeup'
    out.mkdir(parents=True, exist_ok=True)
    char_human_closeup_temporal_weights_in = WEIGHTS_OUT_PATH / "exp_001_overfit_one_weights_checkpoint.0200.json"
    temporal_sequence_command = [
        AF_WRAPPER_PATH,
        "python",
        TRAINER_APP_PATH,
        "--mode apply",
        f"--input {char_human_closeup_temporal_sequence}",
        f"--output {out}",
        f"--weights-in {char_human_closeup_temporal_weights_in}",
    ]
    temporal_sequence_command = " ".join(map(str, temporal_sequence_command))


    job_config = JobConfig(
        name=job_name,
        command_blocks=[
            CommandBlock(
                title="Training Block",
                commands=[
                    Command(
                        title="Train Model",
                        command=train_command
                    )
                ]
            ),
            CommandBlock(
                title="Test Block",
                commands=[
                    Command(
                        title="Spatial Validation Set",
                        command=spatial_validation_command
                    ),
                    Command(
                        title="Sequence over Epochs",
                        command=seqence_over_epoch_command
                    ),
                    Command(
                        title="Temporal Sequence TGB1004140 char_human_closeup",
                        command=temporal_sequence_command
                    )
                ]
            )
        ]
    )

    submit_job(job_config)

if __name__ == "__main__":
    submit()
    launch_tensorboard(TENSORBOARD_LOG_DIR, port=6006)

