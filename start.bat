@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM FARNSWORTH COLLECTIVE - QUICK START SCRIPT (Windows)
REM ═══════════════════════════════════════════════════════════════════════════════
REM
REM Usage:
REM   start.bat           - Run interactive setup and start server
REM   start.bat setup     - Run setup wizard only
REM   start.bat server    - Start server only (assumes .env exists)
REM   start.bat docker    - Build and run with Docker
REM
REM ═══════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo.
echo ╔═══════════════════════════════════════════════════════════════════════════════╗
echo ║                     FARNSWORTH COLLECTIVE - QUICK START                       ║
echo ║          'Good news everyone! Resistance is futile... and delicious!'         ║
echo ╚═══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is required but not installed.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Navigate to script directory
cd /d "%~dp0"

REM Parse arguments
if "%1"=="setup" goto :setup
if "%1"=="server" goto :server
if "%1"=="docker" goto :docker
if "%1"=="help" goto :help
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help

REM Default: setup if needed, then run server
if not exist ".env" (
    echo.
    echo No .env file found. Let's set up your Farnsworth instance!
    echo.
    python setup_farnsworth.py
)

echo.
echo Starting Farnsworth web server...
echo Open: http://localhost:8080
echo.
set PYTHONPATH=%cd%
python -m farnsworth.web.server
goto :end

:setup
echo.
echo Running setup wizard...
echo.
python setup_farnsworth.py
goto :end

:server
if not exist ".env" (
    echo Warning: .env file not found. Running setup first...
    python setup_farnsworth.py
)
echo.
echo Starting Farnsworth web server...
echo.
set PYTHONPATH=%cd%
python -m farnsworth.web.server
goto :end

:docker
if not exist ".env" (
    echo Warning: .env file not found. Running setup first...
    python setup_farnsworth.py
)
echo.
echo Building and starting Docker containers...
echo.
docker-compose up -d --build
echo.
echo [OK] Farnsworth is running!
echo     Web interface: http://localhost:8080
echo     Logs: docker-compose logs -f
goto :end

:help
echo Usage: start.bat [OPTION]
echo.
echo Options:
echo   (none)    Run setup if needed, then start server
echo   setup     Run interactive setup wizard only
echo   server    Start web server only (requires .env)
echo   docker    Build and run with Docker Compose
echo   help      Show this help message
goto :end

:end
endlocal
