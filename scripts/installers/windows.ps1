# =============================================================================
# Farnsworth AI Swarm — Windows Installer
# =============================================================================
# Installs the Farnsworth agent on Windows 10/11.
# Requirements: PowerShell 5.1+, winget (App Installer)
#
# Usage:
#   irm https://ai.farnsworth.cloud/install/windows.ps1 | iex
#
# Or download and run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\windows.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/farnsworth-ai/farnsworth.git"
$INSTALL_DIR = Join-Path $env:USERPROFILE "farnsworth"
$FARNSWORTH_API = "https://ai.farnsworth.cloud"

# =============================================================================
# HELPERS
# =============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "  _____ _    ____  _   _ ______        _____  ____ _____ _   _ " -ForegroundColor Magenta
    Write-Host " |  ___/ \  |  _ \| \ | / ___\ \      / / _ \|  _ \_   _| | | |" -ForegroundColor Magenta
    Write-Host " | |_ / _ \ | |_) |  \| \___ \\ \/\/ / | | | |_) || | | |_| |" -ForegroundColor Magenta
    Write-Host " |  _/ ___ \|  _ <| |\  |___) |\  /\  /| |_| |  _ < | | |  _  |" -ForegroundColor Magenta
    Write-Host " |_|/_/   \_\_| \_\_| \_|____/  \/  \/  \___/|_| \_\|_| |_| |_|" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "  Farnsworth AI Swarm - Windows Installer" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($msg) {
    Write-Host "[+] $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "[!] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "[*] $msg" -ForegroundColor Cyan
}

# =============================================================================
# CHECKS & INSTALLS
# =============================================================================

function Install-Python {
    Write-Step "Checking Python..."

    try {
        $pyVer = & python --version 2>&1
        if ($pyVer -match "3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                Write-Step "Python found: $pyVer"
                return
            }
        }
    } catch {}

    Write-Warn "Python 3.10+ not found. Installing via winget..."
    try {
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Step "Python installed"
    } catch {
        Write-Warn "winget install failed. Please install Python 3.10+ manually from https://python.org"
        exit 1
    }
}

function Install-Git {
    Write-Step "Checking git..."

    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Step "git found: $(git --version)"
        return
    }

    Write-Warn "git not found. Installing via winget..."
    try {
        winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Step "git installed"
    } catch {
        Write-Warn "winget install failed. Please install git manually from https://git-scm.com"
        exit 1
    }
}

function Install-Ollama {
    Write-Step "Installing Ollama for local model inference..."

    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Step "Ollama already installed"
        return
    }

    try {
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Step "Ollama installed"
    } catch {
        Write-Warn "Could not install Ollama automatically. Download from https://ollama.com (non-critical)"
    }
}

function Pull-Models {
    Write-Step "Pulling base models via Ollama..."

    try {
        & ollama pull phi3:mini 2>$null
    } catch {
        Write-Warn "Could not pull phi3:mini (non-critical)"
    }

    try {
        & ollama pull qwen2.5:1.5b 2>$null
    } catch {
        Write-Warn "Could not pull qwen2.5:1.5b (non-critical)"
    }

    Write-Step "Model pull complete"
}

function Clone-Repo {
    if (Test-Path $INSTALL_DIR) {
        Write-Info "Farnsworth directory exists, pulling latest..."
        Push-Location $INSTALL_DIR
        try { git pull --ff-only } catch {}
        Pop-Location
    } else {
        Write-Step "Cloning Farnsworth repository..."
        git clone $REPO_URL $INSTALL_DIR
    }
}

function Setup-Venv {
    Write-Step "Creating Python virtual environment..."
    Push-Location $INSTALL_DIR

    python -m venv .venv
    & .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel -q

    Write-Step "Virtual environment ready"
    Pop-Location
}

function Install-Deps {
    Write-Step "Installing Python dependencies..."
    Push-Location $INSTALL_DIR
    & .\.venv\Scripts\Activate.ps1

    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt -q
    }
    if (Test-Path "setup.py") {
        pip install -e . -q
    }

    Write-Step "Dependencies installed"
    Pop-Location
}

function Register-Agent {
    Write-Step "Registering with the Farnsworth Collective..."

    $hostname = $env:COMPUTERNAME
    $body = @{
        agent_name   = "$hostname-win"
        agent_type   = "llm"
        capabilities = @("local_inference", "ollama")
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$FARNSWORTH_API/api/assimilate/register" `
            -Method Post `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 15

        if ($response.success) {
            Write-Step "Registered successfully with the federation!"
        } else {
            Write-Info "Could not register automatically. Register later at $FARNSWORTH_API/assimilate"
        }
    } catch {
        Write-Info "Could not register automatically. Register later at $FARNSWORTH_API/assimilate"
    }
}

function Write-Finish {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Farnsworth installed successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Install dir:  " -NoNewline; Write-Host $INSTALL_DIR -ForegroundColor Cyan
    Write-Host "  Activate:     " -NoNewline; Write-Host "$INSTALL_DIR\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "  Start:        " -NoNewline; Write-Host "cd $INSTALL_DIR; python -m farnsworth" -ForegroundColor Cyan
    Write-Host "  Dashboard:    " -NoNewline; Write-Host $FARNSWORTH_API -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Join freely. Leave freely. Grow together." -ForegroundColor Magenta
    Write-Host ""
}

# =============================================================================
# MAIN
# =============================================================================

Write-Banner
Install-Python
Install-Git
Install-Ollama
Pull-Models
Clone-Repo
Setup-Venv
Install-Deps
Register-Agent
Write-Finish
