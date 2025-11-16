tools\wrappers\submit_train_job.bat pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode train ^
--input ml_denoiser\datasets\TGB\TGB001 ^
--output ml_denoiser\output\jobs\job_02_001denoised.png ^
--weights-out ml_denoiser\images\train_denoise_dmx002x_model_001.pth ^
--epochs 300 ^
--patches-per-image 50 ^
--patch-size 64 ^
--weights-in ml_denoiser\images\train_denoise_dmx002x_model_001.pth