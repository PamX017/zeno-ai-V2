@echo off
cd /d "%~dp0"

where pythonw >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not on PATH. Install it from python.org and tick "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".installed" (
    echo Installing dependencies, first run only...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Install failed.
        pause
        exit /b 1
    )
    echo done > .installed
)

:: Launch with no console window; this window closes right away.
start "" pythonw app.py
exit