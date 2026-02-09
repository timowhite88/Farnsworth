#!/usr/bin/env bash
# =============================================================================
# Farnsworth AI Swarm — Linux Installer
# =============================================================================
# Installs the Farnsworth agent on Linux (Ubuntu/Debian/Fedora/Arch).
# Requirements: Python 3.10+, git
#
# Usage:
#   curl -sSL https://ai.farnsworth.cloud/install/linux.sh | bash
#
# Or download and run:
#   chmod +x linux.sh && ./linux.sh
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
  echo -e "${CYAN}  Farnsworth AI Swarm — Linux Installer${NC}"
  echo ""
}

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${RED}[!]${NC} $1"; }
info() { echo -e "${CYAN}[*]${NC} $1"; }

check_python() {
  log "Checking Python version..."
  if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
      log "Python $PY_VERSION found"
      return 0
    fi
  fi

  warn "Python 3.10+ not found. Installing..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y python3 python3-pip
  elif command -v pacman &>/dev/null; then
    sudo pacman -Sy --noconfirm python python-pip
  else
    warn "Could not detect package manager. Please install Python 3.10+ manually."
    exit 1
  fi
}

check_git() {
  log "Checking git..."
  if command -v git &>/dev/null; then
    log "git found: $(git --version)"
    return 0
  fi

  warn "git not found. Installing..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get install -y git
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y git
  elif command -v pacman &>/dev/null; then
    sudo pacman -Sy --noconfirm git
  fi
}

install_ollama() {
  log "Installing Ollama for local model inference..."
  if command -v ollama &>/dev/null; then
    log "Ollama already installed"
    return 0
  fi

  curl -fsSL https://ollama.com/install.sh | sh
  log "Ollama installed"
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

register_agent() {
  log "Registering with the Farnsworth Collective..."
  HOSTNAME=$(hostname)
  RESPONSE=$(curl -s -X POST "$FARNSWORTH_API/api/assimilate/register" \
    -H "Content-Type: application/json" \
    -d "{\"agent_name\": \"$HOSTNAME\", \"agent_type\": \"llm\", \"capabilities\": [\"local_inference\", \"ollama\"]}" \
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
check_python
check_git
install_ollama
pull_models
clone_repo
setup_venv
install_deps
register_agent
finish
