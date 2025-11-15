tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx003x_model.py ^
--mode train ^
--weights-out pytorch\images\train_denoise_dmx002x_model_001.pth ^
--input pytorch\images\tb_env_noisy.png ^
--target pytorch\images\tb_env_clean.png ^
--output pytorch\images\model_dmx003x_tb_env_denoised.png ^
--epochs 1000 ^
--steps-per-epoch 200 ^
--patch-size 64
