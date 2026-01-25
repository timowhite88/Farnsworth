@echo off
echo Initializing Farnsworth Environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.10+
    pause
    exit /b
)

REM Create Virtual Environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM Install Dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install Ollama if not present (User guidance)
echo.
echo IMPORTANT: Ensure Ollama is installed and running.
echo Please run: ollama pull llama3.1
echo.

echo Setup Complete. activating environment...
cmd /k "venv\Scripts\activate"
