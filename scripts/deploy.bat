@echo off
REM Farnsworth Deploy Script for Windows
REM Run this on the server to pull updates and restart

echo ==========================================
echo   FARNSWORTH DEPLOYMENT SCRIPT
echo ==========================================

cd /d %~dp0..
echo Working directory: %CD%

REM 1. Pull latest changes
echo.
echo [1/5] Pulling latest changes from GitHub...
git fetch origin
git pull origin main
echo Done.

REM 2. Set up environment variables
echo.
echo [2/5] Configuring environment...

if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
)

REM Check if BANKR_API_KEY exists in .env
findstr /C:"BANKR_API_KEY" .env >nul 2>&1
if errorlevel 1 (
    echo.>> .env
    echo # Bankr Trading Engine>> .env
    echo BANKR_API_KEY=bk_77UE569TAXFUR7ZRYPQUYLS45T4R7S4V>> .env
    echo BANKR_ENABLED=true>> .env
    echo BANKR_DEFAULT_CHAIN=base>> .env
    echo BANKR_TRADING_ENABLED=true>> .env
    echo BANKR_MAX_TRADE_USD=1000.00>> .env
    echo BANKR_POLYMARKET_ENABLED=true>> .env
    echo Bankr API configured.
) else (
    echo Bankr API already configured.
)

REM Check if X402_ENABLED exists in .env
findstr /C:"X402_ENABLED" .env >nul 2>&1
if errorlevel 1 (
    echo.>> .env
    echo # x402 Micropayments>> .env
    echo X402_ENABLED=true>> .env
    echo X402_NETWORK=base>> .env
    echo x402 configured.
) else (
    echo x402 already configured.
)

REM 3. Install dependencies
echo.
echo [3/5] Checking dependencies...
py -m pip install -r requirements.txt --quiet 2>nul
echo Dependencies checked.

REM 4. Restart server
echo.
echo [4/5] Restarting Farnsworth server...

REM Kill existing Python processes running main.py
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Farnsworth*" 2>nul
taskkill /F /IM py.exe /FI "WINDOWTITLE eq Farnsworth*" 2>nul

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start server
if not exist logs mkdir logs
start "Farnsworth Server" /B py main.py > logs\farnsworth.log 2>&1
echo Server starting...

REM Wait for server
timeout /t 5 /nobreak >nul

REM 5. Announce update
echo.
echo [5/5] Announcing update to swarm...

REM Use PowerShell to send the announcement
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:8080/api/swarm/inject' -Method POST -ContentType 'application/json' -Body '{\"bot_name\": \"Farnsworth\", \"content\": \"🚀 **MAJOR UPDATE DEPLOYED** - Bankr Trading, x402 Payments, NLP Tasks, Desktop App, Browser Agent, Web IDE, UE5 + CAD Integration. Try: Hey Farn, what is the price of Bitcoin?\"}' | Out-Null; Write-Host 'Update announced to swarm' } catch { Write-Host 'Could not announce (server may still be starting)' }"

echo.
echo ==========================================
echo   DEPLOYMENT COMPLETE
echo ==========================================
echo.
echo New capabilities:
echo   - Bankr Trading Engine
echo   - x402 Micropayments
echo   - Natural Language Tasks
echo   - Desktop Interface
echo   - Agentic Browser
echo   - Web IDE
echo   - UE5 Integration
echo   - CAD Integration
echo.
echo Server logs: type logs\farnsworth.log
echo.
pause
