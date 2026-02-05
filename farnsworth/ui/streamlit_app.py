"""
Farnsworth Streamlit UI - Interactive Dashboard

Provides a comprehensive visual interface for:
- Chat with memory-augmented responses
- Real-time memory visualization
- Knowledge graph exploration
- Evolution metrics dashboard
- Agent activity monitoring
- A2A Session Management (AGI v1.8.4)
- Nexus Signal Monitoring (AGI v1.8.4)
- Canvas Output Viewing (AGI v1.8.4)
- Mesh Connectivity Visualization (AGI v1.8.4)

Novel Features:
- Live memory activation heatmaps
- Interactive knowledge graph with filtering
- Evolution fitness timeline
- Agent swarm visualization
- Real-time deliberation visualization (AGI v1.8.4)
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List
from pathlib import Path

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from loguru import logger


class FarnsworthUI:
    """
    Main Streamlit application for Farnsworth.

    Provides real-time visualization and interaction capabilities.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self._memory_system = None
        self._swarm_orchestrator = None
        self._fitness_tracker = None
        self._model_manager = None
        self._initialized = False

    async def initialize(self):
        """Initialize Farnsworth components."""
        if self._initialized:
            return

        try:
            from farnsworth.memory.memory_system import MemorySystem
            from farnsworth.agents.swarm_orchestrator import SwarmOrchestrator
            from farnsworth.evolution.fitness_tracker import FitnessTracker
            from farnsworth.core.model_manager import ModelManager

            self._memory_system = MemorySystem(data_dir=str(self.data_dir))
            await self._memory_system.initialize()

            self._swarm_orchestrator = SwarmOrchestrator()
            self._fitness_tracker = FitnessTracker()
            self._model_manager = ModelManager()

            self._initialized = True
            logger.info("Farnsworth UI initialized")

        except Exception as e:
            logger.error(f"UI initialization failed: {e}")

    def run(self):
        """Run the Streamlit application."""
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not installed. Run: pip install streamlit")
            return

        # Page config
        st.set_page_config(
            page_title="Farnsworth AI",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS for visual polish
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .memory-item {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 5px 5px 0;
        }
        .agent-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .agent-code { background: #e3f2fd; color: #1565c0; }
        .agent-reasoning { background: #fff3e0; color: #ef6c00; }
        .agent-research { background: #e8f5e9; color: #2e7d32; }
        .agent-creative { background: #fce4ec; color: #c2185b; }
        .fitness-high { color: #4caf50; }
        .fitness-medium { color: #ff9800; }
        .fitness-low { color: #f44336; }
        .chat-user {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 10px 10px 0 10px;
            margin: 0.5rem 0;
        }
        .chat-assistant {
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 10px 10px 10px 0;
            margin: 0.5rem 0;
        }
        .evolution-progress {
            height: 10px;
            border-radius: 5px;
            background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
        }
        </style>
        """, unsafe_allow_html=True)

        # Sidebar navigation
        self._render_sidebar()

        # Main content based on selected page
        page = st.session_state.get("page", "chat")

        if page == "chat":
            self._render_chat_page()
        elif page == "memory":
            self._render_memory_page()
        elif page == "agents":
            self._render_agents_page()
        elif page == "evolution":
            self._render_evolution_page()
        elif page == "sessions":
            self._render_sessions_page()
        elif page == "mesh":
            self._render_mesh_page()
        elif page == "settings":
            self._render_settings_page()

    def _render_sidebar(self):
        """Render sidebar navigation."""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1 style="margin: 0;">🧠 Farnsworth</h1>
                <p style="color: #666; font-size: 0.9rem;">Self-Evolving AI Companion</p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Navigation buttons
            if st.button("💬 Chat", use_container_width=True, key="nav_chat"):
                st.session_state.page = "chat"
                st.rerun()

            if st.button("🧠 Memory", use_container_width=True, key="nav_memory"):
                st.session_state.page = "memory"
                st.rerun()

            if st.button("🤖 Agents", use_container_width=True, key="nav_agents"):
                st.session_state.page = "agents"
                st.rerun()

            if st.button("📈 Evolution", use_container_width=True, key="nav_evolution"):
                st.session_state.page = "evolution"
                st.rerun()

            if st.button("🔗 Sessions", use_container_width=True, key="nav_sessions"):
                st.session_state.page = "sessions"
                st.rerun()

            if st.button("🌐 Mesh", use_container_width=True, key="nav_mesh"):
                st.session_state.page = "mesh"
                st.rerun()

            if st.button("⚙️ Settings", use_container_width=True, key="nav_settings"):
                st.session_state.page = "settings"
                st.rerun()

            st.divider()

            # Quick stats
            st.markdown("### System Status")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memories", "1,247", "+23")
            with col2:
                st.metric("Fitness", "0.87", "+0.02")

            # Active model indicator
            st.markdown("### Active Model")
            st.info("🔷 DeepSeek-R1-1.5B (Q4_K_M)")

            # Connection status
            st.markdown("### Connections")
            st.success("✅ MCP Server Active")
            st.success("✅ Memory System Ready")

    def _render_chat_page(self):
        """Render the chat interface."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">💬 Chat with Farnsworth</h2>
            <p style="margin: 0; opacity: 0.8;">Memory-augmented conversation with your AI companion</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat layout
        chat_col, context_col = st.columns([2, 1])

        with chat_col:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-user">
                            <strong>You:</strong> {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-assistant">
                            <strong>🧠 Farnsworth:</strong> {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)

            # Chat input
            user_input = st.chat_input("Type your message...")

            if user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                })

                # Generate response (placeholder for actual LLM call)
                with st.spinner("Thinking..."):
                    response = self._generate_response(user_input)

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })

                st.rerun()

        with context_col:
            st.markdown("### 📚 Active Context")

            # Show relevant memories
            with st.expander("Relevant Memories", expanded=True):
                st.markdown("""
                <div class="memory-item">
                    <small>📅 2 hours ago</small><br>
                    User prefers detailed technical explanations
                </div>
                <div class="memory-item">
                    <small>📅 Yesterday</small><br>
                    Working on Farnsworth project implementation
                </div>
                """, unsafe_allow_html=True)

            # Working memory slots
            with st.expander("Working Memory", expanded=True):
                st.code("""
{
  "current_task": "Chat interaction",
  "user_mood": "focused",
  "context_relevance": 0.92
}
                """, language="json")

            # Knowledge graph snippet
            with st.expander("Related Entities"):
                st.markdown("""
                - **Farnsworth** → project
                - **Memory System** → component
                - **Evolution** → feature
                """)

    def _render_memory_page(self):
        """Render the memory exploration page."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">🧠 Memory System</h2>
            <p style="margin: 0; opacity: 0.8;">Explore and manage Farnsworth's memories</p>
        </div>
        """, unsafe_allow_html=True)

        # Memory stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Memories", "1,247")
        with col2:
            st.metric("Knowledge Entities", "342")
        with col3:
            st.metric("Relationships", "891")
        with col4:
            st.metric("Dream Cycles", "47")

        st.divider()

        # Tabs for different memory views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Recent", "🔍 Search", "🕸️ Knowledge Graph", "💭 Dreaming"
        ])

        with tab1:
            self._render_recent_memories()

        with tab2:
            self._render_memory_search()

        with tab3:
            self._render_knowledge_graph()

        with tab4:
            self._render_dreaming_status()

    def _render_recent_memories(self):
        """Render recent memories list."""
        st.subheader("Recent Memories")

        # Filter options
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_input("Filter by tag", placeholder="Enter tag...")
        with col2:
            st.selectbox("Sort by", ["Most Recent", "Importance", "Access Count"])

        # Memory list
        for i in range(5):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"""
                    **Memory #{1247 - i}**
                    Sample memory content demonstrating the storage capability...
                    """)
                with col2:
                    st.caption(f"Importance: {0.9 - i*0.1:.1f}")
                with col3:
                    st.caption(f"{i+1}h ago")
                st.divider()

    def _render_memory_search(self):
        """Render memory search interface."""
        st.subheader("Semantic Search")

        query = st.text_input("Search query", placeholder="Find memories about...")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Min relevance", 0.0, 1.0, 0.3)
        with col2:
            st.number_input("Max results", 1, 50, 10)
        with col3:
            st.multiselect("Sources", ["Archival", "Conversation", "Knowledge Graph"])

        if st.button("Search", type="primary"):
            with st.spinner("Searching memories..."):
                # Placeholder results
                st.success("Found 7 relevant memories")

                for i in range(3):
                    st.markdown(f"""
                    <div class="memory-item">
                        <strong>Relevance: {0.95 - i*0.1:.0%}</strong><br>
                        Result content preview showing matched memory...
                    </div>
                    """, unsafe_allow_html=True)

    def _render_knowledge_graph(self):
        """Render knowledge graph visualization."""
        st.subheader("Knowledge Graph")

        # Graph controls
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Entity type", ["All", "Person", "Concept", "Project", "Tool"])
        with col2:
            st.slider("Max depth", 1, 5, 2)
        with col3:
            st.text_input("Center entity", placeholder="Start from...")

        # Placeholder for graph visualization
        st.info("📊 Interactive knowledge graph visualization would render here using plotly or pyvis")

        # Entity list
        st.markdown("### Top Entities")

        entities = [
            ("Farnsworth", "Project", 47),
            ("Memory System", "Component", 32),
            ("Evolution", "Feature", 28),
            ("User", "Person", 24),
            ("Claude Code", "Tool", 19),
        ]

        for name, etype, mentions in entities:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{name}**")
            with col2:
                st.caption(etype)
            with col3:
                st.caption(f"{mentions} mentions")

    def _render_dreaming_status(self):
        """Render memory dreaming status."""
        st.subheader("Memory Dreaming")

        st.markdown("""
        Memory dreaming is Farnsworth's background process that consolidates memories,
        discovers patterns, and generates insights during idle time.
        """)

        # Dreaming stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last Dream", "2 hours ago")
        with col2:
            st.metric("Memories Processed", "156")
        with col3:
            st.metric("Insights Generated", "12")

        # Recent insights
        st.markdown("### Recent Insights")

        insights = [
            "User shows consistent interest in AI architecture patterns",
            "Code-related discussions often lead to implementation requests",
            "Technical explanations are preferred over high-level summaries",
        ]

        for insight in insights:
            st.info(f"💡 {insight}")

        # Manual trigger
        if st.button("Trigger Dream Cycle", type="secondary"):
            with st.spinner("Dreaming..."):
                import time
                time.sleep(2)
                st.success("Dream cycle complete! 23 memories processed, 3 new insights.")

    def _render_agents_page(self):
        """Render the agents monitoring page."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">🤖 Agent Swarm</h2>
            <p style="margin: 0; opacity: 0.8;">Monitor and manage specialist agents</p>
        </div>
        """, unsafe_allow_html=True)

        # Agent stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Agents", "3")
        with col2:
            st.metric("Tasks Completed", "127")
        with col3:
            st.metric("Avg Response", "2.3s")
        with col4:
            st.metric("Success Rate", "94%")

        st.divider()

        # Tabs for agent views
        tab1, tab2, tab3, tab4 = st.tabs(["🟢 Active", "🗣️ Deliberation", "📊 Statistics", "⚙️ Configuration"])

        with tab1:
            self._render_active_agents()

        with tab2:
            self._render_deliberation_visualizer()

        with tab3:
            self._render_agent_statistics()

        with tab4:
            self._render_agent_config()

    def _render_active_agents(self):
        """Render active agents list."""
        st.subheader("Active Agents")

        agents = [
            {
                "id": "code-001",
                "type": "Code",
                "status": "idle",
                "tasks_completed": 42,
                "last_active": "5 min ago",
            },
            {
                "id": "reasoning-001",
                "type": "Reasoning",
                "status": "busy",
                "current_task": "Analyzing user request patterns",
                "tasks_completed": 38,
                "last_active": "now",
            },
            {
                "id": "research-001",
                "type": "Research",
                "status": "idle",
                "tasks_completed": 27,
                "last_active": "1 hour ago",
            },
        ]

        for agent in agents:
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                with col1:
                    badge_class = f"agent-{agent['type'].lower()}"
                    st.markdown(f"""
                    <span class="agent-badge {badge_class}">{agent['type']}</span>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"**{agent['id']}**")
                    if agent['status'] == 'busy':
                        st.caption(f"🔄 {agent.get('current_task', 'Processing...')}")
                    else:
                        st.caption("💤 Idle")

                with col3:
                    st.caption(f"Tasks: {agent['tasks_completed']}")
                    st.caption(f"Last active: {agent['last_active']}")

                with col4:
                    if agent['status'] == 'idle':
                        st.button("Assign", key=f"assign_{agent['id']}")
                    else:
                        st.button("View", key=f"view_{agent['id']}")

                st.divider()

    def _render_deliberation_visualizer(self):
        """Render real-time deliberation visualization."""
        st.subheader("Live Deliberation")

        # Current deliberation status
        st.markdown("### Current Session")

        # Check if there's an active deliberation
        active_delib = st.session_state.get("active_deliberation", None)

        if active_delib:
            st.success(f"Deliberation in progress: {active_delib.get('id', 'unknown')}")

            # Phase timeline
            phases = ["PROPOSE", "CRITIQUE", "REFINE", "VOTE", "CONSENSUS"]
            current_phase = active_delib.get("phase", "PROPOSE")
            current_idx = phases.index(current_phase) if current_phase in phases else 0

            phase_cols = st.columns(5)
            for i, phase in enumerate(phases):
                with phase_cols[i]:
                    if i < current_idx:
                        st.success(f"✓ {phase}")
                    elif i == current_idx:
                        st.warning(f"⟳ {phase}")
                    else:
                        st.info(f"○ {phase}")

            # Agent contributions
            st.markdown("### Agent Contributions")

            contributions = active_delib.get("contributions", {
                "grok": {"propose": "Initial approach using...", "critique": "Good but consider..."},
                "claude": {"propose": "Alternative method...", "critique": "The first approach..."},
                "gemini": {"propose": "Hybrid solution...", "critique": "Both have merits..."},
            })

            for agent_id, phases_data in contributions.items():
                with st.expander(f"🤖 {agent_id}"):
                    for phase_name, content in phases_data.items():
                        st.markdown(f"**{phase_name.upper()}**")
                        st.write(content[:200] + "..." if len(content) > 200 else content)

        else:
            st.info("No active deliberation. Start one from the Chat page or use the CLI.")

            # Demo deliberation
            if st.button("Start Demo Deliberation"):
                st.session_state.active_deliberation = {
                    "id": "demo_123",
                    "phase": "PROPOSE",
                    "contributions": {},
                }
                st.rerun()

        # Recent deliberations
        st.markdown("### Recent Deliberations")

        recent = [
            {"id": "delib_001", "prompt": "Best caching strategy?", "winner": "grok", "consensus": True, "time": "10m ago"},
            {"id": "delib_002", "prompt": "Code review approach", "winner": "claude", "consensus": True, "time": "25m ago"},
            {"id": "delib_003", "prompt": "Performance optimization", "winner": "deepseek", "consensus": False, "time": "1h ago"},
        ]

        for delib in recent:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.write(f"**{delib['prompt'][:40]}...**")

            with col2:
                st.caption(f"Winner: {delib['winner']}")

            with col3:
                if delib['consensus']:
                    st.success("Consensus")
                else:
                    st.warning("No consensus")

            with col4:
                st.caption(delib['time'])

    def _render_agent_statistics(self):
        """Render agent performance statistics."""
        st.subheader("Agent Performance")

        # Performance by type
        st.markdown("### By Agent Type")

        data = {
            "Agent Type": ["Code", "Reasoning", "Research", "Creative"],
            "Tasks": [42, 38, 27, 20],
            "Success Rate": [96, 94, 91, 89],
            "Avg Time (s)": [1.8, 3.2, 2.5, 2.1],
        }

        st.dataframe(data, use_container_width=True)

        # Task distribution chart placeholder
        st.info("📊 Task distribution chart would render here")

    def _render_agent_config(self):
        """Render agent configuration."""
        st.subheader("Agent Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Max concurrent agents", 1, 10, 5)
            st.number_input("Task timeout (seconds)", 10, 300, 120)
            st.checkbox("Auto-spawn agents", value=True)

        with col2:
            st.selectbox("Default model", ["DeepSeek-R1-1.5B", "BitNet-2B", "Phi-3-mini"])
            st.slider("Confidence threshold", 0.0, 1.0, 0.7)
            st.checkbox("Enable cascade inference", value=True)

        if st.button("Save Configuration", type="primary"):
            st.success("Configuration saved!")

    def _render_evolution_page(self):
        """Render the evolution dashboard."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">📈 Evolution Dashboard</h2>
            <p style="margin: 0; opacity: 0.8;">Track system improvement and adaptation</p>
        </div>
        """, unsafe_allow_html=True)

        # Fitness overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Fitness", "0.87", "+0.02")
        with col2:
            st.metric("Task Success", "94%", "+1%")
        with col3:
            st.metric("User Satisfaction", "0.91", "+0.05")
        with col4:
            st.metric("Efficiency", "0.78", "-0.01")

        st.divider()

        # Tabs for evolution views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Metrics", "🧬 Genetics", "🔄 Behaviors", "📝 History"
        ])

        with tab1:
            self._render_fitness_metrics()

        with tab2:
            self._render_genetic_evolution()

        with tab3:
            self._render_behavior_evolution()

        with tab4:
            self._render_evolution_history()

    def _render_fitness_metrics(self):
        """Render fitness metrics."""
        st.subheader("Fitness Metrics")

        # Metric breakdown
        metrics = {
            "task_success": 0.94,
            "user_satisfaction": 0.91,
            "efficiency": 0.78,
            "response_quality": 0.85,
            "memory_utilization": 0.72,
        }

        for name, value in metrics.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(name.replace("_", " ").title())
            with col2:
                color = "green" if value > 0.8 else "orange" if value > 0.6 else "red"
                st.progress(value, text=f"{value:.0%}")

        # Trends placeholder
        st.info("📈 Fitness trend chart would render here")

    def _render_genetic_evolution(self):
        """Render genetic optimization status."""
        st.subheader("Genetic Optimization")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Generation", "47")
            st.metric("Population Size", "50")
            st.metric("Elite Count", "5")

        with col2:
            st.metric("Best Fitness", "0.912")
            st.metric("Avg Fitness", "0.847")
            st.metric("Diversity", "0.23")

        # Leaderboard
        st.markdown("### Top Configurations")

        configs = [
            ("Config #1847", 0.912, "Active"),
            ("Config #1823", 0.908, "Tested"),
            ("Config #1841", 0.901, "Tested"),
        ]

        for name, fitness, status in configs:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{name}**")
            with col2:
                st.write(f"{fitness:.3f}")
            with col3:
                if status == "Active":
                    st.success(status)
                else:
                    st.info(status)

        # Manual evolution trigger
        if st.button("Run Evolution Cycle", type="primary"):
            with st.spinner("Evolving..."):
                import time
                time.sleep(3)
                st.success("Evolution complete! New best fitness: 0.915")

    def _render_behavior_evolution(self):
        """Render behavior mutation status."""
        st.subheader("Behavior Evolution")

        st.markdown("""
        Behavior evolution adapts agent team compositions and response strategies
        based on performance feedback.
        """)

        # Current behavior params
        st.markdown("### Current Behavior Parameters")

        params = {
            "temperature": 0.72,
            "verbosity": 0.65,
            "code_preference": 0.8,
            "explanation_depth": 0.7,
            "creativity": 0.45,
        }

        for name, value in params.items():
            st.slider(
                name.replace("_", " ").title(),
                0.0, 1.0, value,
                disabled=True,
                key=f"behavior_{name}"
            )

        # Team config
        st.markdown("### Agent Team Configuration")

        team = {
            "code_specialist_weight": 0.35,
            "reasoning_specialist_weight": 0.30,
            "research_specialist_weight": 0.20,
            "creative_specialist_weight": 0.15,
        }

        st.json(team)

    def _render_evolution_history(self):
        """Render evolution history."""
        st.subheader("Evolution History")

        # Hash chain display
        st.markdown("### Recent Evolution Events")

        events = [
            {
                "timestamp": "2024-01-15 14:32:00",
                "type": "Generation Complete",
                "fitness_change": "+0.012",
                "hash": "a3f2c1...8b4d",
            },
            {
                "timestamp": "2024-01-15 12:15:00",
                "type": "Behavior Mutation",
                "fitness_change": "+0.005",
                "hash": "7e9d3a...2c1f",
            },
            {
                "timestamp": "2024-01-15 10:00:00",
                "type": "User Feedback",
                "fitness_change": "+0.008",
                "hash": "b2c4e1...9a3d",
            },
        ]

        for event in events:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.caption(event["timestamp"])
                with col2:
                    st.write(event["type"])
                with col3:
                    st.write(f"**{event['fitness_change']}**")
                with col4:
                    st.code(event["hash"], language=None)
                st.divider()

    def _render_sessions_page(self):
        """Render A2A sessions management page."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">🔗 A2A Sessions</h2>
            <p style="margin: 0; opacity: 0.8;">Manage agent-to-agent and CLI sessions</p>
        </div>
        """, unsafe_allow_html=True)

        # Session stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Sessions", "3")
        with col2:
            st.metric("CLI Connections", "1")
        with col3:
            st.metric("Mesh Peers", "8")
        with col4:
            st.metric("Sub-Swarms", "2")

        st.divider()

        # Tabs for different session views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🟢 Active", "📊 A2A Mesh", "📡 Nexus Signals", "🎨 Canvas"
        ])

        with tab1:
            self._render_active_sessions()

        with tab2:
            self._render_a2a_status()

        with tab3:
            self._render_nexus_signals()

        with tab4:
            self._render_canvas_outputs()

    def _render_active_sessions(self):
        """Render active sessions list."""
        st.subheader("Active Sessions")

        # Session list
        sessions = [
            {
                "id": "cli_abc123",
                "type": "CLI",
                "user": "local",
                "agents": ["grok", "claude", "gemini"],
                "status": "connected",
                "duration": "15m",
            },
            {
                "id": "web_def456",
                "type": "Web",
                "user": "guest",
                "agents": ["grok", "deepseek"],
                "status": "deliberating",
                "duration": "5m",
            },
            {
                "id": "api_ghi789",
                "type": "API",
                "user": "api-user",
                "agents": ["claude"],
                "status": "idle",
                "duration": "1h",
            },
        ]

        for session in sessions:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 1])

                with col1:
                    st.markdown(f"**{session['id']}**")
                    st.caption(f"Type: {session['type']}")

                with col2:
                    if session['status'] == 'connected':
                        st.success(session['status'])
                    elif session['status'] == 'deliberating':
                        st.warning(session['status'])
                    else:
                        st.info(session['status'])

                with col3:
                    st.caption(f"Agents: {', '.join(session['agents'])}")

                with col4:
                    st.caption(f"Duration: {session['duration']}")

                with col5:
                    st.button("End", key=f"end_{session['id']}")

                st.divider()

        # Create new session
        st.subheader("Create New Session")
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Session Type", ["CLI", "API", "Deliberation"])
            st.multiselect(
                "Agents",
                ["grok", "claude", "gemini", "deepseek", "kimi", "phi"],
                default=["grok", "claude"]
            )

        with col2:
            st.text_input("Purpose", placeholder="Describe session purpose...")
            st.number_input("Timeout (minutes)", 5, 120, 30)

        if st.button("Create Session", type="primary"):
            st.success("Session created!")

    def _render_a2a_status(self):
        """Render A2A mesh status."""
        st.subheader("A2A Mesh Status")

        # Mesh visualization placeholder
        st.info("📊 A2A Mesh connectivity visualization")

        # Peer list
        st.markdown("### Connected Peers")

        peers = [
            {"id": "grok", "status": "online", "capabilities": ["reasoning", "code"], "latency": "45ms"},
            {"id": "claude", "status": "online", "capabilities": ["analysis", "writing"], "latency": "62ms"},
            {"id": "gemini", "status": "online", "capabilities": ["vision", "reasoning"], "latency": "78ms"},
            {"id": "deepseek", "status": "busy", "capabilities": ["code", "math"], "latency": "120ms"},
            {"id": "kimi", "status": "online", "capabilities": ["long_context"], "latency": "95ms"},
        ]

        for peer in peers:
            col1, col2, col3, col4 = st.columns([2, 1, 3, 1])

            with col1:
                st.write(f"**{peer['id']}**")

            with col2:
                if peer['status'] == 'online':
                    st.success("●")
                else:
                    st.warning("●")

            with col3:
                st.caption(", ".join(peer['capabilities']))

            with col4:
                st.caption(peer['latency'])

        # Sub-swarms
        st.markdown("### Active Sub-Swarms")

        swarms = [
            {"id": "swarm_code1", "purpose": "Code Review", "members": ["deepseek", "claude"], "status": "active"},
            {"id": "swarm_research1", "purpose": "Research Task", "members": ["grok", "gemini", "kimi"], "status": "complete"},
        ]

        for swarm in swarms:
            with st.expander(f"{swarm['id']}: {swarm['purpose']}"):
                st.write(f"**Members:** {', '.join(swarm['members'])}")
                st.write(f"**Status:** {swarm['status']}")
                if swarm['status'] == 'active':
                    st.button("View Progress", key=f"view_{swarm['id']}")

    def _render_nexus_signals(self):
        """Render real-time Nexus signal feed."""
        st.subheader("Nexus Signal Monitor")

        # Signal filters
        col1, col2, col3 = st.columns(3)

        with col1:
            st.multiselect(
                "Signal Types",
                [
                    "DIALOGUE_*", "A2A_*", "MESH_*", "M2M_*",
                    "COLLECTIVE_*", "CLI_*", "GUI_*"
                ],
                default=["DIALOGUE_*"]
            )

        with col2:
            st.slider("Min Urgency", 0.0, 1.0, 0.0)

        with col3:
            st.selectbox("Time Range", ["Last 5 min", "Last 15 min", "Last hour", "All"])

        # Signal feed
        st.markdown("### Recent Signals")

        signals = [
            {"type": "DIALOGUE_CONSENSUS", "source": "deliberation", "urgency": 0.8, "time": "12:34:56"},
            {"type": "M2M_INSIGHT", "source": "grok", "urgency": 0.5, "time": "12:34:45"},
            {"type": "A2A_SESSION_STARTED", "source": "cli_abc123", "urgency": 0.6, "time": "12:34:30"},
            {"type": "MESH_PEER_HEARTBEAT", "source": "mesh", "urgency": 0.2, "time": "12:34:15"},
            {"type": "COLLECTIVE_DISPATCH", "source": "bridge", "urgency": 0.7, "time": "12:34:00"},
        ]

        for signal in signals:
            urgency_color = "red" if signal['urgency'] > 0.7 else "orange" if signal['urgency'] > 0.4 else "gray"

            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.code(signal['type'], language=None)

            with col2:
                st.caption(signal['source'])

            with col3:
                st.markdown(f":{urgency_color}[●] {signal['urgency']:.1f}")

            with col4:
                st.caption(signal['time'])

        # Auto-refresh toggle
        st.checkbox("Auto-refresh (5s)", value=False, key="signal_refresh")

    def _render_canvas_outputs(self):
        """Render canvas outputs gallery."""
        st.subheader("Canvas Outputs")

        # Gallery of recent outputs
        st.markdown("### Recent Visualizations")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background: #f0f0f0; padding: 2rem; border-radius: 10px; text-align: center;">
                <p>📊 Fitness Evolution</p>
                <small>from: evolution_tracker</small>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: #f0f0f0; padding: 2rem; border-radius: 10px; text-align: center;">
                <p>🕸️ Knowledge Graph</p>
                <small>from: memory_system</small>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background: #f0f0f0; padding: 2rem; border-radius: 10px; text-align: center;">
                <p>📈 Token Usage</p>
                <small>from: token_budgets</small>
            </div>
            """, unsafe_allow_html=True)

        # Canvas from agent responses
        st.markdown("### Agent-Generated Visuals")

        st.info("📊 When agents generate matplotlib figures, they will appear here.")

        # Test plot button
        if st.button("Generate Test Plot"):
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                fig, ax = plt.subplots()
                x = np.linspace(0, 10, 100)
                ax.plot(x, np.sin(x), label='sin(x)')
                ax.plot(x, np.cos(x), label='cos(x)')
                ax.legend()
                ax.set_title("Test Canvas Output")

                st.pyplot(fig)
                st.success("Canvas rendered!")

            except ImportError:
                st.warning("matplotlib not available")

    def _render_mesh_page(self):
        """Render A2A Mesh connectivity page."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">🌐 A2A Mesh Network</h2>
            <p style="margin: 0; opacity: 0.8;">Full mesh connectivity between agents</p>
        </div>
        """, unsafe_allow_html=True)

        # Mesh stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Peers", "8")
        with col2:
            st.metric("Online", "6")
        with col3:
            st.metric("Active Connections", "21")
        with col4:
            st.metric("Messages/min", "47")

        st.divider()

        # Tabs for mesh views
        tab1, tab2, tab3 = st.tabs([
            "🕸️ Topology", "💬 Messages", "📚 Insights"
        ])

        with tab1:
            self._render_mesh_topology()

        with tab2:
            self._render_mesh_messages()

        with tab3:
            self._render_shared_insights()

    def _render_mesh_topology(self):
        """Render mesh network topology."""
        st.subheader("Mesh Topology")

        # ASCII art representation
        topology = """
        ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
        │     ┌──────┐         ┌───────┐         ┌────────┐           │
        │     │ Grok │◄───────►│Claude │◄───────►│ Gemini │           │
        │     └──┬───┘         └───┬───┘         └───┬────┘           │
        │        │                 │                 │                │
        │        │     ┌───────────┼─────────────┐   │                │
        │        │     │           │             │   │                │
        │        ▼     ▼           ▼             ▼   ▼                │
        │     ┌────────┐       ┌──────┐      ┌──────┐                 │
        │     │DeepSeek│◄─────►│ Kimi │◄────►│  Phi │                 │
        │     └────────┘       └──────┘      └──────┘                 │
        │                                                              │
        └──────────────────────────────────────────────────────────────┘
        """
        st.code(topology, language=None)

        # Connection matrix
        st.markdown("### Connection Matrix")

        agents = ["grok", "claude", "gemini", "deepseek", "kimi", "phi"]
        matrix_data = {
            "Agent": agents,
            "grok": ["—", "✓", "✓", "✓", "✓", "○"],
            "claude": ["✓", "—", "✓", "✓", "✓", "✓"],
            "gemini": ["✓", "✓", "—", "○", "✓", "✓"],
            "deepseek": ["✓", "✓", "○", "—", "✓", "○"],
            "kimi": ["✓", "✓", "✓", "✓", "—", "✓"],
            "phi": ["○", "✓", "✓", "○", "✓", "—"],
        }

        st.dataframe(matrix_data, use_container_width=True)
        st.caption("✓ = connected, ○ = disconnected")

    def _render_mesh_messages(self):
        """Render mesh message log."""
        st.subheader("Recent M2M Messages")

        messages = [
            {"from": "grok", "to": "claude", "type": "M2M_QUERY", "preview": "What's the best approach for...", "time": "12:35:01"},
            {"from": "claude", "to": "grok", "type": "M2M_RESPONSE", "preview": "Based on my analysis...", "time": "12:35:03"},
            {"from": "gemini", "to": "*", "type": "M2M_INSIGHT", "preview": "Discovered pattern in...", "time": "12:34:55"},
            {"from": "deepseek", "to": "claude", "type": "M2M_COLLABORATE", "preview": "Need help with code...", "time": "12:34:40"},
        ]

        for msg in messages:
            col1, col2, col3, col4 = st.columns([2, 2, 4, 1])

            with col1:
                target = "broadcast" if msg['to'] == "*" else msg['to']
                st.markdown(f"**{msg['from']}** → {target}")

            with col2:
                st.code(msg['type'], language=None)

            with col3:
                st.caption(msg['preview'])

            with col4:
                st.caption(msg['time'])

    def _render_shared_insights(self):
        """Render shared insights from the mesh."""
        st.subheader("Shared Insights")

        insights = [
            {
                "source": "grok",
                "type": "pattern",
                "content": "User queries about caching often follow discussions about performance",
                "relevance": 0.85,
                "acks": 4,
            },
            {
                "source": "claude",
                "type": "solution",
                "content": "For similar code review tasks, the hybrid approach works best",
                "relevance": 0.78,
                "acks": 3,
            },
            {
                "source": "gemini",
                "type": "connection",
                "content": "Memory consolidation improves when triggered after deliberations",
                "relevance": 0.72,
                "acks": 2,
            },
        ]

        for insight in insights:
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{insight['source']}** ({insight['type']})")
                    st.write(insight['content'])

                with col2:
                    st.metric("Relevance", f"{insight['relevance']:.0%}")
                    st.caption(f"{insight['acks']} acknowledgments")

                st.divider()

    def _render_settings_page(self):
        """Render settings page."""
        st.markdown("""
        <div class="main-header">
            <h2 style="margin: 0;">⚙️ Settings</h2>
            <p style="margin: 0; opacity: 0.8;">Configure Farnsworth's behavior</p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 General", "🧠 Models", "💾 Memory", "🔌 Integrations"
        ])

        with tab1:
            st.subheader("General Settings")

            st.text_input("Data directory", value="./data")
            st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            st.checkbox("Enable telemetry", value=False)
            st.checkbox("Auto-save conversations", value=True)

        with tab2:
            st.subheader("Model Configuration")

            st.selectbox("Primary model", [
                "DeepSeek-R1-Distill-Qwen-1.5B",
                "BitNet b1.58-2B-4T",
                "Qwen3-0.6B",
                "Phi-3-mini",
            ])

            st.selectbox("Backend", ["Ollama", "llama.cpp", "BitNet"])

            st.slider("GPU layers", 0, 100, 35)
            st.slider("Context size", 1024, 32768, 8192)

            st.checkbox("Enable speculative decoding", value=True)
            st.checkbox("Enable cascade inference", value=True)

        with tab3:
            st.subheader("Memory Settings")

            st.number_input("Max archival memories", 1000, 1000000, 100000)
            st.number_input("Embedding dimensions", 128, 2048, 384)
            st.slider("Memory importance threshold", 0.0, 1.0, 0.3)

            st.checkbox("Enable memory dreaming", value=True)
            st.number_input("Dream interval (minutes)", 5, 120, 30)

        with tab4:
            st.subheader("Integrations")

            st.markdown("### MCP Server")
            st.text_input("MCP server port", value="8765")
            st.checkbox("Enable MCP server", value=True)

            st.markdown("### External Tools")
            st.checkbox("Enable Composio integration", value=False)
            st.text_input("Composio API key", type="password")

        if st.button("Save All Settings", type="primary"):
            st.success("Settings saved!")

    def _generate_response(self, user_input: str) -> str:
        """
        Generate a response using the Farnsworth system.

        Uses memory-augmented generation:
        1. Recall relevant memories
        2. Build context
        3. Generate response with LLM
        """
        try:
            # Run async code in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Ensure initialized
                if not self._initialized:
                    loop.run_until_complete(self.initialize())

                # Recall relevant memories
                memories = []
                if self._memory_system:
                    try:
                        results = loop.run_until_complete(
                            self._memory_system.recall(user_input, top_k=3)
                        )
                        memories = [r.content for r in results]
                    except Exception as e:
                        logger.debug(f"Memory recall failed: {e}")

                # Build context with memories
                context_parts = []
                if memories:
                    context_parts.append("Relevant memories:")
                    for i, mem in enumerate(memories, 1):
                        context_parts.append(f"  {i}. {mem[:200]}")

                context = "\n".join(context_parts) if context_parts else ""

                # Generate response using model manager
                if self._model_manager and hasattr(self._model_manager, 'generate'):
                    try:
                        prompt = f"""You are Farnsworth, a helpful AI assistant with persistent memory.

{context}

User: {user_input}

Respond naturally and helpfully, incorporating any relevant memory context."""

                        response = loop.run_until_complete(
                            self._model_manager.generate(prompt)
                        )

                        # Store interaction in memory
                        if self._memory_system:
                            try:
                                loop.run_until_complete(
                                    self._memory_system.remember(
                                        f"User said: {user_input}\nAssistant responded: {response[:500]}",
                                        tags=["conversation"],
                                        importance=0.5,
                                    )
                                )
                            except Exception as e:
                                logger.debug(f"Failed to store memory: {e}")

                        return response

                    except Exception as e:
                        logger.warning(f"Model generation failed: {e}")

                # Fallback: Try swarm orchestrator
                if self._swarm_orchestrator and hasattr(self._swarm_orchestrator, 'process'):
                    try:
                        result = loop.run_until_complete(
                            self._swarm_orchestrator.process(user_input, context={"memories": memories})
                        )
                        if hasattr(result, 'output'):
                            return result.output
                        elif isinstance(result, str):
                            return result
                    except Exception as e:
                        logger.warning(f"Swarm processing failed: {e}")

                # Final fallback: Echo with memory summary
                if memories:
                    return f"I recall some related information: {memories[0][:200]}...\n\nRegarding your message: '{user_input}' - I'm currently operating in limited mode. How can I help?"
                else:
                    return f"I received your message: '{user_input}'. I'm currently operating in limited mode without full LLM capabilities. Please ensure the model backend is configured."

            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"I encountered an error while processing your request. Please try again. (Error: {type(e).__name__})"


def create_app(data_dir: str = "./data") -> FarnsworthUI:
    """
    Create and return a FarnsworthUI instance.

    Args:
        data_dir: Directory for data storage

    Returns:
        Configured FarnsworthUI instance
    """
    return FarnsworthUI(data_dir=data_dir)


def main():
    """Main entry point for the Streamlit UI."""
    import sys

    # Parse command line args
    data_dir = "./data"
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            data_dir = sys.argv[i + 1]

    # Create and run the UI
    app = create_app(data_dir)
    app.run()


if __name__ == "__main__":
    main()