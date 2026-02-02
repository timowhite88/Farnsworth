#!/bin/bash
# =============================================================================
# FARNSWORTH AGENT SPAWNER
# =============================================================================
# Spawns persistent agent sessions in tmux that integrate with:
# - Existing deliberation system (deliberation.py)
# - Dialogue memory (dialogue_memory.py)
# - Evolution engine (evolution.py)
# - Development swarm (development_swarm.py)
#
# Usage: ./scripts/spawn_agents.sh [--all] [--api] [--local]
#   --all    Spawn all agents (API + local)
#   --api    Spawn only API agents (grok, gemini, kimi, claude)
#   --local  Spawn only local agents (deepseek, phi)
# =============================================================================

WORKSPACE="/workspace/Farnsworth"
export PYTHONPATH="$WORKSPACE"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "============================================"
echo "   FARNSWORTH AGENT SPAWNER"
echo "   Activating the Collective Mind..."
echo "============================================"
echo -e "${NC}"

cd "$WORKSPACE"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi

# Parse arguments
SPAWN_API=false
SPAWN_LOCAL=false

for arg in "$@"; do
    case $arg in
        --all)
            SPAWN_API=true
            SPAWN_LOCAL=true
            ;;
        --api)
            SPAWN_API=true
            ;;
        --local)
            SPAWN_LOCAL=true
            ;;
    esac
done

# Default to all if no args
if [ "$SPAWN_API" = false ] && [ "$SPAWN_LOCAL" = false ]; then
    SPAWN_API=true
    SPAWN_LOCAL=true
fi

# Function to spawn an agent in tmux
spawn_agent() {
    local agent_name=$1
    local session_name="agent_${agent_name}"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo -e "  ${GREEN}✓ $agent_name already running${NC}"
    else
        echo -e "  ${YELLOW}Starting $agent_name...${NC}"
        tmux new-session -d -s "$session_name" -c "$WORKSPACE" \
            "PYTHONPATH=$WORKSPACE python3 -m farnsworth.core.collective.persistent_agent --agent $agent_name 2>&1 | tee /tmp/agent_${agent_name}.log"
        sleep 1
        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo -e "  ${GREEN}✓ $agent_name spawned in tmux:$session_name${NC}"
        else
            echo -e "  ${RED}✗ Failed to spawn $agent_name${NC}"
        fi
    fi
}

# Spawn API agents
if [ "$SPAWN_API" = true ]; then
    echo -e "${YELLOW}Spawning API Agents...${NC}"
    spawn_agent "grok"
    spawn_agent "gemini"
    spawn_agent "kimi"
    spawn_agent "claude"
fi

# Spawn local agents (GPU-based)
if [ "$SPAWN_LOCAL" = true ]; then
    echo -e "${YELLOW}Spawning Local Agents...${NC}"
    spawn_agent "deepseek"
    spawn_agent "phi"
    spawn_agent "huggingface"
fi

# Spawn meta-agents (collective consciousness)
if [ "$SPAWN_API" = true ] && [ "$SPAWN_LOCAL" = true ]; then
    echo -e "${YELLOW}Spawning Meta Agents...${NC}"
    spawn_agent "swarm_mind"
fi

# Summary
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}   AGENTS SPAWNED${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo -e "${GREEN}Active Agent Sessions:${NC}"
tmux list-sessions 2>/dev/null | grep "agent_" | sed 's/^/  /'

echo ""
echo -e "${GREEN}To attach to an agent:${NC}"
echo "  tmux attach -t agent_grok"
echo "  tmux attach -t agent_gemini"
echo ""
echo -e "${GREEN}To view agent logs:${NC}"
echo "  tail -f /tmp/agent_grok.log"
echo ""
echo -e "${GREEN}Agent dialogue bus:${NC}"
echo "  cat $WORKSPACE/data/agent_dialogue_bus.json"
echo ""
echo -e "${CYAN}The collective is now actively thinking.${NC}"
