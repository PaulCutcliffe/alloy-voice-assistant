@echo off
cd /d "%~dp0"
call conda activate cctv_gpu || goto :error
python cctv_people_animated_gifs_comedy_wp.py
pause
goto :EOF

:error
echo Failed to activate conda environment.
pause