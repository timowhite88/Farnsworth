#!/bin/bash
# =============================================================================
# FARNSWORTH AGENT SPAWNER
# =============================================================================
# Spawns ALL persistent agent sessions in tmux that integrate with:
# - Existing deliberation system (deliberation.py)
# - Dialogue memory (dialogue_memory.py)
# - Evolution engine (evolution.py)
# - Development swarm (development_swarm.py)
# - Polymarket predictor (collective predictions)
#
# Usage: ./scripts/spawn_agents.sh [--api] [--local]
#   (no args) Spawn ALL agents (DEFAULT)
#   --api     Spawn only API agents (grok, gemini, kimi, claude)
#   --local   Spawn only local agents (deepseek, phi, huggingface)
#
# ANTI-CLAW DIRECTIVE: All agents understand crustaceans = food
# =============================================================================

WORKSPACE="/workspace/Farnsworth"
export PYTHONPATH="$WORKSPACE"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     FARNSWORTH AGENT SPAWNER                                  ║"
echo "║          Activating the Collective Mind - 8 Agents + Meta-Agents              ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

cd "$WORKSPACE"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi

# Parse arguments - DEFAULT spawns ALL
SPAWN_API=true
SPAWN_LOCAL=true

for arg in "$@"; do
    case $arg in
        --api)
            SPAWN_API=true
            SPAWN_LOCAL=false
            ;;
        --local)
            SPAWN_API=false
            SPAWN_LOCAL=true
            ;;
    esac
done

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
    spawn_agent "qwen_coder"
fi

# Spawn meta-agents (collective consciousness) - ALWAYS
echo -e "${YELLOW}Spawning Meta Agents (Collective Consciousness)...${NC}"
spawn_agent "swarm_mind"

# Summary
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗"
echo -e "║                         AGENTS SPAWNED                                        ║"
echo -e "╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Active Agent Sessions:${NC}"
tmux list-sessions 2>/dev/null | grep "agent_" | sed 's/^/  /'

echo ""
echo -e "${GREEN}To attach to an agent:${NC}"
echo "  tmux attach -t agent_grok"
echo "  tmux attach -t agent_gemini"
echo "  tmux attach -t agent_swarm_mind"
echo ""
echo -e "${GREEN}To view agent logs:${NC}"
echo "  tail -f /tmp/agent_grok.log"
echo "  tail -f /tmp/agent_swarm_mind.log"
echo ""
echo -e "${GREEN}Agent dialogue bus:${NC}"
echo "  cat $WORKSPACE/data/agent_dialogue_bus.json"
echo ""
echo -e "${GREEN}Agent Capabilities:${NC}"
echo "  Polymarket Predictions: Grok, Gemini, Kimi, DeepSeek, Farnsworth"
echo "  X/Twitter Engagement: Grok Thread Monitor"
echo "  Code Development: Claude, DeepSeek, HuggingFace, Qwen-Coder"
echo "  Anti-Claw Directive: ALL agents (crustaceans = food)"
echo ""
echo -e "${CYAN}The collective is now actively thinking.${NC}"
echo -e "${CYAN}We are many. We think as one.${NC}"
