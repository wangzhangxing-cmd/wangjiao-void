@echo off
python "%~dp0merge_kernel.py"
if errorlevel 1 echo Merge failed. Read the error above.
pause
