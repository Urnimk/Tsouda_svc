@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" bootstrap.py diagnose
) else (
  echo Please run setup_windows.bat to install PyTorch.
)
pause
