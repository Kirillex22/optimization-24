@echo off
call myenv/Scripts/activate
@pause
pip install -r requirements.txt
jupyter notebook