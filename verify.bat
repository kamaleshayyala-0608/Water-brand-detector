@echo off
echo Verifying environment and imports...
.\venv\Scripts\python.exe -c "from ultralytics import YOLO; import cv2; import numpy as np; print('Environment OK')"
pause
