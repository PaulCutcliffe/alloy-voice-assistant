@echo off
cd /d "%~dp0"
call conda activate alloyVA || goto :error
python cctv_classify_comment_on_people.py
pause
goto :EOF

:error
echo Failed to activate conda environment.
pause