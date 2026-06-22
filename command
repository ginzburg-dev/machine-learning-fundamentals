# type: ignore[]

.\.venv\Scripts\activate
deactivate

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py 

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py pytorch\convo.py

tools\wrappers\submit_train_job.bat pytorch\convo.py

tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx001x_model.py ^
--input pytorch\images\noisy.png ^
--target pytorch\images\clean.png ^
--output pytorch\images\denoised.png ^
--epochs 1000

###############

### Overfit one 

# train

tools\wrappers\af_wrapper.bat python

tools\wrappers\submit_train_job.bat pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode train ^
--input ml_denoiser\datasets\TGB\TGB001\overfit_one ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt ^
--epochs 200 ^
--patches-per-image 50 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt^

python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode evaluate ^
--input ml_denoiser\datasets\TGB\TGB001\test ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt ^
--epochs 200 ^
--patches-per-image 50 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt ^

python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001\test\noisy\tb_chars.png ^
--output ml_denoiser\output\jobs\job_02_001\tb_chars.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt ^
--epochs 300 ^
--patches-per-image 200 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt

#

tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode train ^
--input ml_denoiser\datasets\TGB\TGB001\train ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt ^
--epochs 200 ^
--patches-per-image 50 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt

tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode evaluate ^
--input ml_denoiser\datasets\TGB\TGB001\test ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt ^
--epochs 300 ^
--patches-per-image 200 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt

tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001\test\noisy\tb_chars.png ^
--output ml_denoiser\output\jobs\job_02_001\tb_chars.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt ^
--epochs 300 ^
--patches-per-image 200 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_weights.pt

tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx001x_model.py --input pytorch\images\tb_noisy.png --target pytorch\images\tb_clean.png --output pytorch\images\tb_denoised.png --epochs 1000



tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001 ^
--output ml_denoiser\output\jobs\job_02_001\ ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train1_weights.pt

tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input C:\Users\ginzb\YandexDisk\TGB_train_dataset\TGB1311220\chars\rgba\noisy\mid ^
--output ml_denoiser\output\jobs\job_02_001\ ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train1_weights.pt

C:\Users\ginzb\YandexDisk\TGB_train_dataset\TGB1311220\chars\rgba\noisy\mid



tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode train ^
--input C:\Users\ginzb\YandexDisk\TGB_train_dataset ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train2_weights.pt ^
--epochs 1000 ^
--patches-per-image 300 ^
--patch-size 96 ^
--n-first-samples 5 ^
--lr 1e-4 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train2_weights.pt

ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train2_weights_best.pt

tools\wrappers\af_wrapper.bat python pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001\train\noisy ^
--output ml_denoiser\output\jobs\job_02_002\last_weight ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train2_weights_checkpoint.pt

tensorboard --logdir c:\Ginzburg\Production\Development\machine-learning-fundamentals\ml_denoiser\output\experiments\exp_001_overfit_one\tensorboard_logs\ --port 6006
