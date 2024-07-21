@echo off
cd /d "%~dp0"
call conda activate cctv_gpu || goto :error
python cctv_people_animated_gifs_comment_wp_multithreaded.py
pause
goto :EOF

:error
echo Failed to activate conda environment.
pause