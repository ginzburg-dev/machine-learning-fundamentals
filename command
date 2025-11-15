# type: ignore[]

.\.venv\Scripts\activate
deactivate

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py 

tools\wrappers\af_farm_wrapper.bat libs\cgru_farm\submit_job.py pytorch\convo.py

tools\wrappers\submit_train_job.bat pytorch\convo.py