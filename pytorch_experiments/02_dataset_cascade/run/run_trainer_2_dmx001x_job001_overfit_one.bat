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
--input ml_denoiser\datasets\TGB\TGB001\overfit_one ^
--output ml_denoiser\output\jobs\job_02_001_denoised.png ^
--weights-out ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt ^
--epochs 200 ^
--patches-per-image 200 ^
--patch-size 64 ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt

rem ==================== EVALUATE ===================
call tools\wrappers\submit_train_job.bat evaluate pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode evaluate ^
--input ml_denoiser\datasets\TGB\TGB001\train ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt

rem ===================== APPLY =====================
call tools\wrappers\submit_train_job.bat apply pytorch_experiments\experements\02_dataset_cascade\trainer_2_dmx001x.py ^
--mode apply ^
--input ml_denoiser\datasets\TGB\TGB001\test\noisy\tb_chars.png ^
--output "%OUTFILE%" ^
--weights-in ml_denoiser\output\jobs\job_02_001\trainer_2_dmx001x_overfit_one_weights.pt

endlocal