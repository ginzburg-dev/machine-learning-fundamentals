import os
from pathlib import Path

WORKING_DIR = Path(__file__).parent.parent
ML_DENOISER_DIR = WORKING_DIR / "ml_denoiser"
AF_WRAPPER_PATH = ML_DENOISER_DIR / "tools\\/wrappers\\/af_wrapper.bat"
TRAINER_APP_PATH = ML_DENOISER_DIR / "training/run_train.py"
TGB_DATASET_DIR = Path("C:/Users/ginzb/YandexDisk/TGB_train_dataset/")