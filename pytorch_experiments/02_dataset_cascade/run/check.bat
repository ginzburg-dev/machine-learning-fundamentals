rem ===================== APPLY =====================
call tools\wrappers\submit_train_job.bat apply pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001 ^
--output ml_denoiser\output\jobs\job_02_001\ ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_train1_weights.pt