# type: ignore[]

.\.venv\Scripts\activate
deactivate

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py 

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py pytorch\convo.py

tools\wrappers\submit_train_job.bat pytorch\convo.py

tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx001x_model.py --input pytorch\images\noisy.png --target pytorch\images\clean.png --output pytorch\images\denoised.png --epochs 1000

tools\wrappers\submit_train_job.bat pytorch\train_denoise_dmx001x_model.py --input pytorch\images\tb_noisy.png --target pytorch\images\tb_clean.png --output pytorch\images\tb_denoised.png --epochs 1000