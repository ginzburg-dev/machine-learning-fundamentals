@echo off
setlocal

rem === versioned output for APPLY ===
set "BASE=ml_denoiser\output\jobs\job_02_001\tb_chars"
set "EXT=.png"

set /a VER=1

:find_next
rem zero-pad to 3 digits: 001, 002, ...
set "PADDED=00%VER%"
set "PADDED=%PADDED:~-3%"

set "OUTFILE=%BASE%_v%PADDED%%EXT%"

if exist "%OUTFILE%" (
    set /a VER+=1
    goto :find_next
)

echo Next free version file: %OUTFILE%

rem ===================== TRAIN =====================
call tools\wrappers\submit_train_job.bat train pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode train ^
--input C:\Users\ginzb\YandexDisk\TGB_train_dataset ^
--output ml_denoiser\output\jobs\job_02_003_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_003\trainer_2_dmx001x_train3_weights.pt ^
--epochs 250 ^
--batch-size 8 ^
--patches-per-image 200 ^
--patch-size 128 ^
--n-first-samples 100 ^
--n-first-frames 3 ^
--lr 1e-4 ^
--weights-in ml_denoiser\output\jobs\job_02_003\trainer_2_dmx001x_train3_weights.pt

rem ==================== EVALUATE ===================
call tools\wrappers\submit_train_job.bat evaluate pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode evaluate ^
--input ml_denoiser\datasets\TGB\TGB001\train ^
--weights-in ml_denoiser\output\jobs\job_02_003\trainer_2_dmx001x_train3_weights.pt

rem ===================== APPLY =====================
call tools\wrappers\submit_train_job.bat apply pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001\train\noisy ^
--output ml_denoiser\output\jobs\job_02_003\ ^
--weights-in ml_denoiser\output\jobs\job_02_003\trainer_2_dmx001x_train3_weights.pt

endlocal

echo Starting TensorBoard...
tensorboard --logdir C:\Ginzburg\Production\Development\machine-learning-fundamentals\ml_denoiser\output\jobs\job_02_003\tensorboard_logs --port 6006
pause