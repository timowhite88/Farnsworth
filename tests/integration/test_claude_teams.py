"""
Integration tests for Claude Teams Fusion (AGI v1.9).

Tests the delegation pipeline, team coordination, MCP bridge,
and error handling when Claude SDK/CLI is unavailable.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import asdict


class TestAgentSDKBridge:
    """Test Claude Agent SDK bridge with fallback chain."""

    def test_import(self):
        """SDK bridge should be importable."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import (
            AgentSDKBridge, ClaudeModel, AgentStatus, AgentResponse, get_sdk_bridge,
        )
        assert AgentSDKBridge is not None

    def test_model_enum(self):
        """ClaudeModel enum should have expected values."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        assert ClaudeModel.SONNET.value == "sonnet"
        assert ClaudeModel.OPUS.value == "opus"
        assert ClaudeModel.HAIKU.value == "haiku"
        assert ClaudeModel.OPUS_4_6.value == "claude-opus-4-6-20260205"

    def test_is_available_with_api_key(self):
        """Bridge should report available when API key is set."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import AgentSDKBridge

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            bridge = AgentSDKBridge.__new__(AgentSDKBridge)
            bridge._sdk_available = False
            bridge._cli_available = False
            bridge.api_key = "test-key"
            bridge.sessions = {}
            bridge.default_model = None

            assert bridge.is_available() is True

    def test_is_available_without_anything(self):
        """Bridge should report unavailable when nothing is configured."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import AgentSDKBridge

        bridge = AgentSDKBridge.__new__(AgentSDKBridge)
        bridge._sdk_available = False
        bridge._cli_available = False
        bridge.api_key = None
        bridge.sessions = {}

        assert bridge.is_available() is False

    @pytest.mark.asyncio
    async def test_spawn_subagent_fallback(self):
        """When Claude is unavailable, should fall back to shadow agents."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import AgentSDKBridge, ClaudeModel

        bridge = AgentSDKBridge.__new__(AgentSDKBridge)
        bridge._sdk_available = False
        bridge._cli_available = False
        bridge.api_key = None
        bridge.sessions = {}
        bridge.default_model = ClaudeModel.HAIKU

        with patch("farnsworth.core.collective.persistent_agent.call_shadow_agent",
                    new_callable=AsyncMock, return_value=("grok", "test response")):
            result = await bridge.spawn_subagent("test task")
            assert result.content == "test response"
            assert "fallback" in result.session_id

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test create -> send -> close session lifecycle."""
        from farnsworth.integration.claude_teams.agent_sdk_bridge import (
            AgentSDKBridge, ClaudeModel, AgentStatus,
        )

        bridge = AgentSDKBridge.__new__(AgentSDKBridge)
        bridge._sdk_available = False
        bridge._cli_available = False
        bridge.api_key = "test"
        bridge.sessions = {}
        bridge.default_model = ClaudeModel.SONNET

        session = await bridge.create_session(system_prompt="Test")
        assert session.session_id in bridge.sessions
        assert session.status == AgentStatus.IDLE

        await bridge.close_session(session.session_id)
        assert session.session_id not in bridge.sessions


class TestSwarmTeamFusion:
    """Test the main orchestration layer."""

    def test_import(self):
        """SwarmTeamFusion should be importable."""
        from farnsworth.integration.claude_teams.swarm_team_fusion import (
            SwarmTeamFusion, DelegationType, OrchestrationMode,
        )
        assert SwarmTeamFusion is not None

    def test_delegation_types(self):
        """All delegation types should be defined."""
        from farnsworth.integration.claude_teams.swarm_team_fusion import DelegationType

        expected = ["research", "analysis", "coding", "critique", "synthesis", "creative", "execution"]
        actual = [d.value for d in DelegationType]
        for exp in expected:
            assert exp in actual, f"Missing delegation type: {exp}"

    def test_orchestration_modes(self):
        """All orchestration modes should be defined."""
        from farnsworth.integration.claude_teams.swarm_team_fusion import OrchestrationMode

        expected = ["sequential", "parallel", "pipeline", "competitive"]
        actual = [m.value for m in OrchestrationMode]
        for exp in expected:
            assert exp in actual, f"Missing orchestration mode: {exp}"

    def test_agent_switches(self):
        """Agent switches should include all expected agents."""
        from farnsworth.integration.claude_teams.swarm_team_fusion import SwarmTeamFusion

        with patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_sdk_bridge"), \
             patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_team_coordinator"), \
             patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_mcp_server"):
            fusion = SwarmTeamFusion()

            switches = fusion.get_agent_switches()
            assert "haiku" in switches
            assert "sonnet" in switches
            assert "opus" in switches
            assert "opus_4_6" in switches
            assert "teams" in switches

    def test_best_available_model(self):
        """get_best_available_model should respect switches and priority."""
        from farnsworth.integration.claude_teams.swarm_team_fusion import SwarmTeamFusion
        from farnsworth.integration.claude_teams.agent_sdk_bridge import ClaudeModel

        with patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_sdk_bridge"), \
             patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_team_coordinator"), \
             patch("farnsworth.integration.claude_teams.swarm_team_fusion.get_mcp_server"):
            fusion = SwarmTeamFusion()

            # Default priority: opus_4_6 first
            best = fusion.get_best_available_model()
            assert best == ClaudeModel.OPUS_4_6

            # Disable opus_4_6, should fall to sonnet
            fusion.set_agent_switch("opus_4_6", False)
            best = fusion.get_best_available_model()
            assert best == ClaudeModel.SONNET


class TestMCPBridge:
    """Test MCP tool registration and access control."""

    def test_import(self):
        """MCP bridge should be importable."""
        from farnsworth.integration.claude_teams.mcp_bridge import (
            FarnsworthMCPServer, MCPToolAccess, MCPTool,
        )
        assert FarnsworthMCPServer is not None

    def test_default_tools_registered(self):
        """Default tools should be registered on init."""
        from farnsworth.integration.claude_teams.mcp_bridge import FarnsworthMCPServer

        server = FarnsworthMCPServer()
        tools = server.list_tools()

        tool_names = [t["name"] for t in tools]
        assert "swarm_oracle_query" in tool_names
        assert "read_swarm_memory" in tool_names
        assert "get_swarm_status" in tool_names
        assert "call_shadow_agent" in tool_names

    def test_access_control(self):
        """Access control should restrict tool usage."""
        from farnsworth.integration.claude_teams.mcp_bridge import (
            FarnsworthMCPServer, MCPToolAccess,
        )

        server = FarnsworthMCPServer()

        # Set team to no access
        server.set_team_access("test_team", MCPToolAccess.NONE)
        assert server.check_access("test_team", "swarm_oracle_query") is False

        # Set team to full access
        server.set_team_access("test_team", MCPToolAccess.FULL)
        assert server.check_access("test_team", "swarm_oracle_query") is True


class TestTeamCoordinator:
    """Test team creation and task management."""

    def test_import(self):
        """TeamCoordinator should be importable."""
        from farnsworth.integration.claude_teams.team_coordinator import (
            TeamCoordinator, TeamRole, TaskPriority, ClaudeTeam,
        )
        assert TeamCoordinator is not None

    def test_team_roles(self):
        """All team roles should be defined."""
        from farnsworth.integration.claude_teams.team_coordinator import TeamRole

        expected = ["lead", "analyst", "developer", "critic", "synthesizer"]
        actual = [r.value for r in TeamRole]
        for exp in expected:
            assert exp in actual

    @pytest.mark.asyncio
    async def test_create_task(self):
        """Should be able to create tasks with priority."""
        from farnsworth.integration.claude_teams.team_coordinator import (
            TeamCoordinator, TaskPriority,
        )

        with patch("farnsworth.integration.claude_teams.team_coordinator.get_sdk_bridge"):
            coordinator = TeamCoordinator()

            task = await coordinator.create_task(
                description="Test task",
                priority=TaskPriority.HIGH,
            )

            assert task.description == "Test task"
            assert task.priority == TaskPriority.HIGH
            assert task.status == "pending"

    def test_stats(self):
        """Stats should return expected structure."""
        from farnsworth.integration.claude_teams.team_coordinator import TeamCoordinator

        with patch("farnsworth.integration.claude_teams.team_coordinator.get_sdk_bridge") as mock_bridge:
            mock_bridge.return_value.get_stats.return_value = {}
            coordinator = TeamCoordinator()

            stats = coordinator.get_stats()
            assert "total_teams" in stats
            assert "active_teams" in stats
            assert "pending_tasks" in stats
            assert "nexus_connected" in stats
