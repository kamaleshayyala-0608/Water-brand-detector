@echo off
echo Check if input test video is present...
if not exist videos\input.mp4 (
    echo input.mp4 missing. Generating synthetic test video...
    .\venv\Scripts\python.exe utils/generate_test_video.py
)
echo.
echo Launching Video Detector...
.\venv\Scripts\python.exe inference/video.py
pause
