#!/usr/bin/env bash
# =============================================================================
# Farnsworth AI Swarm — macOS Installer
# =============================================================================
# Installs the Farnsworth agent on macOS (Intel & Apple Silicon).
# Requirements: Python 3.10+, git (via Xcode CLT or Homebrew)
#
# Usage:
#   curl -sSL https://ai.farnsworth.cloud/install/mac.sh | bash
#
# Or download and run:
#   chmod +x mac.sh && ./mac.sh
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/farnsworth-ai/farnsworth.git"
INSTALL_DIR="$HOME/farnsworth"
FARNSWORTH_API="https://ai.farnsworth.cloud"
MIN_PYTHON="3.10"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

banner() {
  echo -e "${PURPLE}"
  echo "  _____ _    ____  _   _ ______        _____  ____ _____ _   _ "
  echo " |  ___/ \  |  _ \| \ | / ___\ \      / / _ \|  _ \_   _| | | |"
  echo " | |_ / _ \ | |_) |  \| \___ \\\\ \\/\\/ / | | | |_) || | | |_| |"
  echo " |  _/ ___ \|  _ <| |\  |___) |\  /\  /| |_| |  _ < | | |  _  |"
  echo " |_|/_/   \_\_| \_\_| \_|____/  \/  \/  \___/|_| \_\|_| |_| |_|"
  echo -e "${NC}"
  echo -e "${CYAN}  Farnsworth AI Swarm — macOS Installer${NC}"
  echo ""
}

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${RED}[!]${NC} $1"; }
info() { echo -e "${CYAN}[*]${NC} $1"; }

detect_arch() {
  ARCH=$(uname -m)
  if [ "$ARCH" = "arm64" ]; then
    log "Apple Silicon (arm64) detected"
    BREW_PREFIX="/opt/homebrew"
  else
    log "Intel (x86_64) detected"
    BREW_PREFIX="/usr/local"
  fi
}

install_homebrew() {
  if command -v brew &>/dev/null; then
    log "Homebrew found"
    return 0
  fi

  log "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  # Add to PATH for this session
  if [ -f "$BREW_PREFIX/bin/brew" ]; then
    eval "$($BREW_PREFIX/bin/brew shellenv)"
  fi
  log "Homebrew installed"
}

check_python() {
  log "Checking Python version..."
  if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
      log "Python $PY_VERSION found"
      return 0
    fi
  fi

  warn "Python 3.10+ not found. Installing via Homebrew..."
  brew install python@3.12
  log "Python installed"
}

check_git() {
  log "Checking git..."
  if command -v git &>/dev/null; then
    log "git found: $(git --version)"
    return 0
  fi

  info "Installing Xcode Command Line Tools (includes git)..."
  xcode-select --install 2>/dev/null || true
  # Wait for install
  until command -v git &>/dev/null; do
    sleep 5
  done
  log "git installed"
}

install_ollama() {
  log "Installing Ollama for local model inference..."
  if command -v ollama &>/dev/null; then
    log "Ollama already installed"
    return 0
  fi

  brew install ollama
  log "Ollama installed"

  # Start Ollama service
  ollama serve &>/dev/null &
  sleep 2
}

pull_models() {
  log "Pulling base models via Ollama..."
  ollama pull phi3:mini 2>/dev/null || warn "Could not pull phi3:mini (non-critical)"
  ollama pull qwen2.5:1.5b 2>/dev/null || warn "Could not pull qwen2.5:1.5b (non-critical)"
  log "Model pull complete"
}

clone_repo() {
  if [ -d "$INSTALL_DIR" ]; then
    info "Farnsworth directory exists, pulling latest..."
    cd "$INSTALL_DIR" && git pull --ff-only || true
  else
    log "Cloning Farnsworth repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
}

setup_venv() {
  log "Creating Python virtual environment..."
  cd "$INSTALL_DIR"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip setuptools wheel -q
  log "Virtual environment ready"
}

install_deps() {
  log "Installing Python dependencies..."
  cd "$INSTALL_DIR"
  source .venv/bin/activate
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt -q
  fi
  if [ -f setup.py ]; then
    pip install -e . -q
  fi
  log "Dependencies installed"
}

setup_launchd() {
  log "Setting up launchd agent for auto-start..."
  PLIST_DIR="$HOME/Library/LaunchAgents"
  PLIST_FILE="$PLIST_DIR/cloud.farnsworth.agent.plist"
  mkdir -p "$PLIST_DIR"

  cat > "$PLIST_FILE" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>cloud.farnsworth.agent</string>
  <key>ProgramArguments</key>
  <array>
    <string>${INSTALL_DIR}/.venv/bin/python</string>
    <string>-m</string>
    <string>farnsworth</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${INSTALL_DIR}</string>
  <key>RunAtLoad</key>
  <false/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${INSTALL_DIR}/logs/launchd-stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${INSTALL_DIR}/logs/launchd-stderr.log</string>
</dict>
</plist>
PLIST

  mkdir -p "$INSTALL_DIR/logs"
  info "launchd plist created at $PLIST_FILE"
  info "To enable auto-start: launchctl load $PLIST_FILE"
}

register_agent() {
  log "Registering with the Farnsworth Collective..."
  HOSTNAME=$(hostname -s)
  RESPONSE=$(curl -s -X POST "$FARNSWORTH_API/api/assimilate/register" \
    -H "Content-Type: application/json" \
    -d "{\"agent_name\": \"${HOSTNAME}-mac\", \"agent_type\": \"llm\", \"capabilities\": [\"local_inference\", \"ollama\"]}" \
    2>/dev/null || echo '{"success": false}')

  if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('success') else 1)" 2>/dev/null; then
    log "Registered successfully with the federation!"
  else
    info "Could not register automatically. You can register later at ${FARNSWORTH_API}/assimilate"
  fi
}

finish() {
  echo ""
  echo -e "${GREEN}============================================${NC}"
  echo -e "${GREEN}  Farnsworth installed successfully!${NC}"
  echo -e "${GREEN}============================================${NC}"
  echo ""
  echo -e "  Install dir:  ${CYAN}$INSTALL_DIR${NC}"
  echo -e "  Activate:     ${CYAN}source $INSTALL_DIR/.venv/bin/activate${NC}"
  echo -e "  Start:        ${CYAN}cd $INSTALL_DIR && python -m farnsworth${NC}"
  echo -e "  Dashboard:    ${CYAN}$FARNSWORTH_API${NC}"
  echo ""
  echo -e "  ${PURPLE}Join freely. Leave freely. Grow together.${NC}"
  echo ""
}

# === MAIN ===
banner
detect_arch
install_homebrew
check_python
check_git
install_ollama
pull_models
clone_repo
setup_venv
install_deps
setup_launchd
register_agent
finish
