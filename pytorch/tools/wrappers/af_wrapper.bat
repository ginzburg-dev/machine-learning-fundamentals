@echo off
REM ---- CGRU / AF env ----
set PIPELINE_TOOLS=C:\Ginzburg\Production\Development\machine-learning-fundamentals\
set CGRU_LOCATION=C:\cgru
set AF_ROOT=%CGRU_LOCATION%\afanasy
set AF_CONFIG=%CGRU_LOCATION%\config
set PATH=%AF_ROOT%\bin;%PATH%;%PIPELINE_TOOLS%
set PYTHONPATH=%AF_ROOT%\python;%PYTHONPATH%
rem Add software to PATH:
SET PATH=%CGRU_LOCATION%\software_setup\bin;%PATH%

rem Python module path:
SET CGRU_PYTHON=%CGRU_LOCATION%\lib\python
if defined PYTHONPATH (
   SET PYTHONPATH=%CGRU_PYTHON%;%PYTHONPATH%
) else (
   SET PYTHONPATH=%CGRU_PYTHON%
)

REM ---- Activate venv ----
call %PIPELINE_TOOLS%\.venv\Scripts\activate.bat

%*