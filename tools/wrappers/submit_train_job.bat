@echo off
REM run_af_farm.bat – called by Afanasy

REM Call torch_wrapper.bat that lives in the same folder as this script
call "%~dp0af_wrapper.bat" python libs\cgru_farm\submit_job.py %*
