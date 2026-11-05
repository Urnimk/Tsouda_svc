@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
if not exist ".venv\Scripts\python.exe" call :create_env
if not exist ".venv\Scripts\python.exe" goto failed
set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 goto failed
"%PY%" -m pip install --upgrade "pip==25.3"
if errorlevel 1 goto failed
"%PY%" -m pip install "torch==2.7.1+cu118" "torchaudio==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto failed
"%PY%" -m pip install --only-binary=:all: -r requirements.txt
if errorlevel 1 goto failed
"%PY%" -m pip check
if errorlevel 1 goto failed
"%PY%" bootstrap.py assets
if errorlevel 1 goto failed
rvc\ffmpeg.exe -version >nul 2>&1
if errorlevel 1 goto failed
rvc\ffprobe.exe -version >nul 2>&1
if errorlevel 1 goto failed
echo ready > ".venv\.tsouda_ready"
"%PY%" bootstrap.py diagnose
if errorlevel 1 echo CUDA test did not pass. CPU mode can be used; see README.md for GPU help.
call start_windows.bat
exit /b %errorlevel%

:create_env
py -3.12-64 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  py -3.12-64 -m venv ".venv"
  exit /b
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" goto local_python
where winget >nul 2>&1
if errorlevel 1 exit /b 1
winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 exit /b 1
if not exist "%LocalAppData%\Programs\Python\Python312\python.exe" exit /b 1
:local_python
"%LocalAppData%\Programs\Python\Python312\python.exe" -m venv ".venv"
exit /b

:failed
echo.
echo Setup failed. Read the first error above and see README.md.
echo Python 3.12 x64 and a working internet connection are required.
pause
exit /b 1
