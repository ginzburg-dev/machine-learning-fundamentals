tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx002x_model.py ^
--input pytorch\images\tb_noisy.png ^
--target pytorch\images\tb_clean.png ^
--output pytorch\images\model_dmx002x_tb_denoised.png ^
--epochs 300 ^
--steps-per-epoch 200 ^
--patch-size 64