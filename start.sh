#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# FARNSWORTH COLLECTIVE - QUICK START SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./start.sh           - Run interactive setup and start server
#   ./start.sh --setup   - Run setup wizard only
#   ./start.sh --server  - Start server only (assumes .env exists)
#   ./start.sh --docker  - Build and run with Docker
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     FARNSWORTH COLLECTIVE - QUICK START                       ║"
echo "║          'Good news everyone! Resistance is futile... and delicious!'         ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
        echo "Please install Python 3.10+ from https://python.org"
        exit 1
    fi
    PYTHON="python"
else
    PYTHON="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
case "${1:-}" in
    --setup)
        echo -e "\n${CYAN}Running setup wizard...${NC}\n"
        $PYTHON setup_farnsworth.py
        ;;
    --server)
        if [ ! -f ".env" ]; then
            echo -e "${YELLOW}Warning: .env file not found. Running setup first...${NC}"
            $PYTHON setup_farnsworth.py
        fi
        echo -e "\n${CYAN}Starting Farnsworth web server...${NC}\n"
        PYTHONPATH="$SCRIPT_DIR" $PYTHON -m farnsworth.web.server
        ;;
    --docker)
        if [ ! -f ".env" ]; then
            echo -e "${YELLOW}Warning: .env file not found. Running setup first...${NC}"
            $PYTHON setup_farnsworth.py
        fi
        echo -e "\n${CYAN}Building and starting Docker containers...${NC}\n"
        docker-compose up -d --build
        echo -e "\n${GREEN}✓${NC} Farnsworth is running!"
        echo -e "   Web interface: http://localhost:8080"
        echo -e "   Logs: docker-compose logs -f"
        ;;
    --help|-h)
        echo "Usage: ./start.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  (none)      Run setup if needed, then start server"
        echo "  --setup     Run interactive setup wizard only"
        echo "  --server    Start web server only (requires .env)"
        echo "  --docker    Build and run with Docker Compose"
        echo "  --help      Show this help message"
        ;;
    *)
        # Default: setup if needed, then run server
        if [ ! -f ".env" ]; then
            echo -e "${YELLOW}No .env file found. Let's set up your Farnsworth instance!${NC}\n"
            $PYTHON setup_farnsworth.py
        fi

        echo -e "\n${CYAN}Starting Farnsworth web server...${NC}"
        echo -e "Open: ${GREEN}http://localhost:8080${NC}\n"
        PYTHONPATH="$SCRIPT_DIR" $PYTHON -m farnsworth.web.server
        ;;
esac
