#!/bin/bash
# ============================================================================
# FARNSWORTH COLLECTIVE - CLAUDE TMUX SESSION STARTER
# ============================================================================
#
# Start Claude Code in a persistent tmux session with MCP memory tools.
#
# Usage:
#   ./scripts/start_claude_tmux.sh
#
# The session will:
# - Create tmux session "farnsworth_claude"
# - Start Claude Code with MCP config if available
# - Load persistent memory
# - Keep session alive for the collective
#
# ============================================================================

set -e

TMUX_SESSION="farnsworth_claude"
WORKSPACE="/workspace/Farnsworth"
MCP_CONFIG="${WORKSPACE}/.mcp/config.json"

echo "=============================================="
echo "FARNSWORTH COLLECTIVE - CLAUDE SESSION"
echo "=============================================="

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed"
    echo "Install with: apt-get install tmux"
    exit 1
fi

# Kill existing session if any
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Killing existing session: $TMUX_SESSION"
    tmux kill-session -t "$TMUX_SESSION"
    sleep 1
fi

# Create new detached session
echo "Creating tmux session: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION"

# Change to workspace directory
tmux send-keys -t "$TMUX_SESSION" "cd $WORKSPACE" Enter
sleep 0.5

# Find Claude Code binary
CLAUDE_BIN=""
if [ -f "$HOME/.local/bin/claude" ]; then
    CLAUDE_BIN="$HOME/.local/bin/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_BIN="claude"
fi

if [ -z "$CLAUDE_BIN" ]; then
    echo "WARNING: Claude Code not found, starting bash shell instead"
    echo "Install Claude Code with: npm install -g @anthropic-ai/claude-code"
    tmux send-keys -t "$TMUX_SESSION" "echo 'Claude Code not installed. Run: npm install -g @anthropic-ai/claude-code'" Enter
else
    # Build Claude command with MCP if config exists
    CLAUDE_CMD="$CLAUDE_BIN"
    if [ -f "$MCP_CONFIG" ]; then
        CLAUDE_CMD="$CLAUDE_BIN --mcp-config $MCP_CONFIG"
        echo "Using MCP config: $MCP_CONFIG"
    fi

    # Start Claude Code
    echo "Starting Claude Code: $CLAUDE_CMD"
    tmux send-keys -t "$TMUX_SESSION" "$CLAUDE_CMD" Enter

    # Wait for Claude to start
    echo "Waiting for Claude to initialize..."
    sleep 5

    # Load persistent memory (if MCP memory command available)
    echo "Loading persistent memory..."
    tmux send-keys -t "$TMUX_SESSION" "/memory load" Enter
    sleep 2
fi

# Verify session is running
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo ""
    echo "=============================================="
    echo "SUCCESS: Claude tmux session started!"
    echo "=============================================="
    echo ""
    echo "Session name: $TMUX_SESSION"
    echo ""
    echo "To attach to session:"
    echo "  tmux attach -t $TMUX_SESSION"
    echo ""
    echo "To view session without attaching:"
    echo "  tmux capture-pane -t $TMUX_SESSION -p"
    echo ""
    echo "To kill session:"
    echo "  tmux kill-session -t $TMUX_SESSION"
    echo ""
else
    echo "ERROR: Failed to create tmux session"
    exit 1
fi
