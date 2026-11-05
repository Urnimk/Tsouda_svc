@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
if not exist ".venv\.tsouda_ready" (
  echo Please run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 (
  echo Launch failed. See README.md for help.
  pause
  exit /b 1
)
