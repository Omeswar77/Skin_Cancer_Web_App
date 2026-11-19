@echo off
echo ======================================================================
echo  Starting DermoAI - Skin Cancer Detection & Dermoscopic Risk Assessment
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please make sure the venv folder exists or recreate it with Python 3.11.
    pause
    exit /b 1
)

echo Activating Python environment...
call venv\Scripts\activate.bat

echo Starting Flask Server on http://127.0.0.1:5000 ...
echo (Keep this window open while using the web application)
echo.
venv\Scripts\python.exe app.py

pause
