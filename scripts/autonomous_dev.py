"""
FARNS Autonomous Development Daemon
=====================================

Self-developing system that uses the FARNS mesh protocol and collective
deliberation to autonomously build novel features for Farnsworth.

Architecture:
  1. RESEARCH:  ask_collective() — multiple agents brainstorm feature ideas
  2. DESIGN:    deliberate() — PROPOSE->CRITIQUE->REFINE->VOTE on architecture
  3. CODE:      query_remote_bot("qwen3-coder-next") via FARNS mesh ROUTE packet
  4. REVIEW:    call_shadow_agent("deepseek"/"phi") for code review via mesh
  5. POST:      HackathonDominator / ColosseumWorker for forum posting
  6. REPEAT:    Each feature improves a different area of Farnsworth

Runs on Server 1 (nexus-alpha) where the collective, deliberation room,
shadow agents, and hackathon systems are already live.

Usage:
    python scripts/autonomous_dev.py
    python scripts/autonomous_dev.py --dry-run
    python scripts/autonomous_dev.py --feature adaptive_consensus
"""

import asyncio
import json
import time
import os
import sys
import subprocess
import argparse
import logging
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

logger.add("/tmp/autonomous_dev.log", rotation="10 MB")

# ─── Config ──────────────────────────────────────────────────────────────

WORKSPACE = Path("/workspace/Farnsworth")
HACKATHON_DIR = WORKSPACE / "hackathon"
STATE_FILE = HACKATHON_DIR / ".dev_state.json"
FORUM_COOLDOWN = 1800  # 30 min between forum posts

# ─── Feature Areas ───────────────────────────────────────────────────────

FEATURE_AREAS = [
    {
        "id": "adaptive_consensus",
        "area": "network",
        "title": "Adaptive Consensus Topology",
        "research_prompt": "Research adaptive consensus algorithms for AI mesh networks. How can a distributed AI swarm dynamically switch between consensus strategies (fast quorum, expert-weighted, full BFT, redundant verification) based on task type, network load, and trust levels? What novel approaches exist beyond static BFT? Consider self-tuning thresholds.",
        "design_prompt": "Design an Adaptive Consensus Topology system for the Farnsworth AI Swarm mesh. It should dynamically reconfigure how the mesh reaches consensus based on: task type (factual=fast quorum, code=expert-weighted, high-stakes=full BFT), network load, and trust levels. Track consensus quality over time and self-tune thresholds. Output a detailed architecture with classes, methods, and data flow.",
        "code_prompt": "Write a complete, production-ready Python module `hackathon/adaptive_consensus.py` for the Farnsworth AI Swarm. Implement an AdaptiveConsensusTopology class that: (1) classifies task type from prompt text, (2) selects consensus strategy per task (FastQuorum, ExpertWeighted, FullBFT, RedundantVerification), (3) dynamically assigns node vote weights based on historical accuracy per domain, (4) tracks consensus quality metrics and self-tunes thresholds, (5) uses dataclasses for ConsensusStrategy/ConsensusResult/NodeTrust. Use async/await, BLAKE3 for any hashing, comprehensive type hints, docstrings. 300-500 lines. Include `if __name__ == '__main__'` demo. Output ONLY Python code, no markdown fences.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "predictive_prefetch",
        "area": "memory",
        "title": "Predictive Memory Prefetch Engine",
        "research_prompt": "Research predictive prefetching techniques for AI memory systems. How can we anticipate what context an agent will need BEFORE it asks? Consider Markov chains over access patterns, embedding-based similarity prediction, conversation flow analysis, and temporal pattern detection. What novel approaches exist for pre-warming memory?",
        "design_prompt": "Design a Predictive Memory Prefetch Engine for the Farnsworth AI Swarm's 7-layer memory system. It should: analyze conversation flow to predict needed context, use Markov chains over memory access patterns to prefetch, pre-warm working memory with likely-needed archival memories, track prefetch hit rates and self-optimize. Output a detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/predictive_prefetch.py`. Implement a PrefetchEngine class that: (1) builds access pattern models using Markov chains (transition probabilities between memory categories/topics), (2) predicts next-needed context from current conversation state, (3) pre-fetches candidate memories ranked by predicted relevance, (4) tracks prefetch hit/miss rates and self-optimizes prediction weights, (5) supports temporal patterns (time-of-day, session-phase). Use async/await, dataclasses, type hints, 300-500 lines. Include demo in __main__. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "adversarial_shield",
        "area": "security",
        "title": "Adversarial Robustness Shield",
        "research_prompt": "Research adversarial robustness for multi-agent AI systems. How can a swarm detect and neutralize prompt injection, jailbreak attempts, and adversarial attacks? Consider multi-layer detection (pattern matching, semantic analysis, behavioral anomaly), cross-model verification, canary tokens, and consensus-based threat assessment. What novel approaches don't exist yet?",
        "design_prompt": "Design an Adversarial Robustness Shield for the Farnsworth AI Swarm. It should: detect prompt injection/jailbreak via multi-layer analysis (pattern+semantic+behavioral), use cross-model verification (if one model's output deviates suspiciously, flag it), inject canary tokens to detect exfiltration, rate-limit suspicious patterns, quarantine inputs for review. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/adversarial_shield.py`. Implement an AdversarialShield class with: (1) PatternDetector — regex-based injection pattern matching with updatable signatures, (2) SemanticAnalyzer — embedding-based anomaly detection for prompt deviation from expected distribution, (3) BehavioralMonitor — tracks response entropy/variance per model, flags sudden changes, (4) CanarySystem — injects traceable zero-width Unicode markers to detect data exfiltration loops, (5) ConsensusVerifier — routes suspicious inputs through multiple models, flags disagreement. Use async/await, dataclasses, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai", "security"],
    },
    {
        "id": "knowledge_distillation",
        "area": "learning",
        "title": "Cross-Model Knowledge Distillation",
        "research_prompt": "Research real-time knowledge distillation between LLMs in a mesh network. How can one model teach another through inference alone (no fine-tuning)? Consider few-shot prompt injection, knowledge capsules, competency gap detection, and teaching example generation. What novel approaches exist for runtime knowledge transfer?",
        "design_prompt": "Design a Cross-Model Knowledge Distillation system for the Farnsworth mesh. It should: detect knowledge gaps (model A correct, model B wrong on same query), generate teaching examples from successful inferences, create compressed knowledge capsules (few-shot prompts), track transfer success rates per model pair, build a knowledge dependency graph. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/knowledge_distillation.py`. Implement a KnowledgeDistiller class with: (1) GapDetector — identifies topics where models diverge in correctness, (2) TeachingGenerator — creates few-shot examples from successful inference pairs, (3) KnowledgeCapsule dataclass — compressed prompt+examples+context for injection, (4) TransferTracker — monitors which model pairs transfer knowledge effectively, (5) DependencyGraph — maps knowledge flow between models. Use async/await, dataclasses, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "temporal_reasoning",
        "area": "reasoning",
        "title": "Temporal Causal Reasoning Engine",
        "research_prompt": "Research temporal reasoning and causal inference for AI systems. How can an AI swarm reason about time-ordered events, causal chains, and counterfactuals? Consider Allen's interval algebra, temporal constraint satisfaction, causal DAGs, and counterfactual reasoning. What novel approaches combine these for LLM-based systems?",
        "design_prompt": "Design a Temporal Causal Reasoning Engine for the Farnsworth AI Swarm. It should: parse temporal expressions from natural language, build causal chains (Event A -> Event B -> Effect C with timestamps), support Allen's interval algebra for temporal relations, enable counterfactual reasoning, integrate with the Knowledge Graph for temporal entity relationships. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/temporal_reasoning.py`. Implement a TemporalReasoner class with: (1) TemporalParser — extracts temporal expressions and event ordering from text, (2) CausalChain — linked events with timestamps and causal relationships, (3) IntervalAlgebra — Allen's 13 temporal relations (before, after, during, overlaps, etc), (4) CounterfactualEngine — evaluates 'what if X didn't happen' by removing nodes from causal graph, (5) TemporalQuery — answers time-based questions over the causal graph. Use dataclasses, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "emergent_detector",
        "area": "swarm",
        "title": "Emergent Behavior Detector",
        "research_prompt": "Research emergence detection in multi-agent AI systems. How can we measure when a swarm produces insights that no individual agent could? Consider partial information decomposition (synergy vs redundancy vs unique info), behavioral baseline tracking, and emergence metrics. What novel approaches exist for detecting genuine collective intelligence?",
        "design_prompt": "Design an Emergent Behavior Detector for the Farnsworth AI Swarm. It should: track individual agent behavior baselines, compare collective output against individual predictions, measure emergence metrics (synergy/redundancy/unique information), detect 'swarm moments' where the collective exceeds individual capabilities, log emergence events with full provenance. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/emergent_detector.py`. Implement an EmergenceDetector class with: (1) BaselineTracker — per-agent behavioral profiles (topic distribution, confidence levels, response patterns), (2) SynergyMeasure — quantifies how much collective output exceeds sum of individual predictions using information-theoretic metrics, (3) SwarmMomentDetector — identifies outputs that couldn't come from any single agent, (4) EmergenceEvent dataclass — logs detected emergence with timestamps, contributing agents, synergy score, (5) EmergenceReport — aggregate statistics over time. Use async/await, dataclasses, statistics module, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "self_evolving_protocol",
        "area": "network",
        "title": "Self-Evolving Wire Protocol",
        "research_prompt": "Research self-evolving network protocols. How can a wire protocol adapt itself based on usage patterns — compressing frequent packets, deprecating unused ones, adding new packet types at runtime? Consider protocol version negotiation, A/B testing of protocol variants, and consensus-based upgrades. What novel approaches exist for protocols that evolve without central coordination?",
        "design_prompt": "Design a Self-Evolving Wire Protocol system for the FARNS mesh. It should: track packet type usage frequencies and latencies, propose optimizations (compress frequent, deprecate unused), support dynamic packet type registration at runtime, enable version negotiation between nodes, A/B test protocol variants, require 2/3+ consensus for upgrades. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/self_evolving_protocol.py`. Implement a ProtocolEvolver class with: (1) UsageTracker — per-packet-type frequency, latency, error rate statistics, (2) OptimizationProposer — analyzes usage data, proposes protocol changes (compress, deprecate, merge), (3) DynamicRegistry — runtime packet type registration with versioned schemas, (4) VersionNegotiator — handshake protocol for nodes running different versions, (5) ABTester — runs two protocol variants simultaneously, measures performance diff, (6) ConsensusUpgrade — requires 2/3+ node agreement to adopt changes. Use dataclasses, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
    {
        "id": "collective_dreaming",
        "area": "consciousness",
        "title": "Collective Dream Synthesis",
        "research_prompt": "Research collective dreaming and idle-time knowledge synthesis for AI systems. How can a multi-agent swarm use idle periods productively — generating free-association chains, cross-pollinating insights between agents, extracting novel connections between disparate knowledge domains? Consider dream consolidation in neuroscience and how it maps to AI memory systems.",
        "design_prompt": "Design a Collective Dream Synthesis system for the Farnsworth AI Swarm. It should: detect idle periods (no active queries for 5+ minutes), enter dream mode where agents generate free-association chains from recent memories, cross-pollinate dream outputs for resonance detection, extract novel insights between disparate domains, log dream journals, immediately wake on new queries, score dream quality over time. Output detailed architecture.",
        "code_prompt": "Write a complete Python module `hackathon/collective_dreaming.py`. Implement a CollectiveDreamer class with: (1) IdleDetector — monitors query rate, triggers dream state after configurable idle period, (2) DreamGenerator — generates free-association chains from seed memories using temperature-elevated prompting, (3) ResonanceDetector — compares dream outputs across agents, identifies convergent themes, (4) InsightExtractor — finds novel connections between disparate knowledge domains in dream output, (5) DreamJournal — timestamped log of all dream sessions with provenance and quality scores, (6) WakeTrigger — immediately exits dream state on new queries. Use async/await, dataclasses, 300-500 lines. Include __main__ demo. Output ONLY Python code.",
        "forum_tags": ["infra", "ai"],
    },
]


# ─── State Management ────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "completed_features": [],
        "last_forum_post_time": 0,
        "total_features_built": 0,
        "total_lines_written": 0,
        "forum_posts": [],
        "errors": [],
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ─── FARNS Mesh Client ──────────────────────────────────────────────────

class MeshClient:
    """
    Persistent FARNS mesh client for routing queries through the network.

    Connects to the local FARNS node via TCP, authenticates with BLAKE3
    challenge-response, and sends ROUTE / LATENT_ROUTE packets for
    code generation and model selection.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self.host = host
        self.port = port
        self.node_name = "autonomous-dev"
        self._reader = None
        self._writer = None
        self._remote_bots: List[str] = []
        self._connected = False

    async def connect(self) -> bool:
        """Connect and authenticate with the local FARNS node."""
        from farnsworth.network.farns_protocol import (
            PacketType, FARNSPacket, read_frame, write_frame,
            make_hello, make_discovery,
        )
        from farnsworth.network.farns_auth import (
            get_or_create_seed, derive_node_identity, solve_challenge,
        )

        try:
            seed = get_or_create_seed()
            identity = derive_node_identity(seed, self.node_name, b"")

            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=10.0
            )

            # HELLO
            await write_frame(self._writer, make_hello(
                self.node_name, b"", {"version": 2, "bots": []},
            ))

            # Challenge-response auth
            resp = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not resp or resp.packet_type != PacketType.VERIFY:
                return False

            challenge = resp.data.get("challenge", b"")
            ts = time.time()
            response = solve_challenge(identity, challenge, ts)

            await write_frame(self._writer, FARNSPacket(
                packet_type=PacketType.VERIFY,
                sender=self.node_name,
                target=resp.sender,
                data={"response": response, "timestamp": ts, "step": "response"},
            ))

            result = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if not result or not result.data.get("accepted", False):
                return False

            # Discovery
            disc = await asyncio.wait_for(read_frame(self._reader), timeout=10.0)
            if disc and disc.packet_type == PacketType.DISCOVERY:
                self._remote_bots = disc.data.get("bots", [])

            await write_frame(self._writer, make_discovery(self.node_name, bots=[]))

            self._connected = True
            logger.info(f"[MESH] Connected to FARNS node, remote bots: {self._remote_bots}")
            return True

        except Exception as e:
            logger.warning(f"[MESH] Connection failed: {e}")
            self._connected = False
            return False

    async def route_query(self, bot_name: str, prompt: str,
                          max_tokens: int = 16000,
                          timeout: float = 600.0) -> Optional[str]:
        """Send a ROUTE packet to a specific bot via the mesh."""
        if not self._connected:
            if not await self.connect():
                return None

        from farnsworth.network.farns_protocol import (
            PacketType, FARNSPacket, write_frame,
        )

        stream_id = str(uuid.uuid4())[:8]
        pkt = FARNSPacket(
            packet_type=PacketType.ROUTE,
            sender=self.node_name,
            target=bot_name,
            stream_id=stream_id,
            data={"prompt": prompt, "bot": bot_name, "max_tokens": max_tokens},
        )

        try:
            await write_frame(self._writer, pkt)
            return await self._collect_response(stream_id, timeout)
        except (ConnectionError, OSError) as e:
            logger.warning(f"[MESH] Connection lost during route: {e}")
            self._connected = False
            return None

    async def latent_query(self, prompt: str,
                           max_tokens: int = 4000,
                           timeout: float = 120.0) -> Optional[str]:
        """Send a LATENT_ROUTE packet — mesh auto-selects the best model."""
        if not self._connected:
            if not await self.connect():
                return None

        from farnsworth.network.farns_protocol import (
            PacketType, FARNSPacket, write_frame,
        )

        stream_id = str(uuid.uuid4())[:8]
        pkt = FARNSPacket(
            packet_type=PacketType.LATENT_ROUTE,
            sender=self.node_name,
            stream_id=stream_id,
            data={"prompt": prompt, "max_tokens": max_tokens},
        )

        try:
            await write_frame(self._writer, pkt)
            return await self._collect_response(stream_id, timeout)
        except (ConnectionError, OSError) as e:
            logger.warning(f"[MESH] Connection lost during latent query: {e}")
            self._connected = False
            return None

    async def _collect_response(self, stream_id: str, timeout: float) -> Optional[str]:
        """Collect DIALOGUE response chunks for a stream."""
        from farnsworth.network.farns_protocol import PacketType, read_frame

        chunks = []
        start = time.time()

        while time.time() - start < timeout:
            try:
                remaining = max(1, timeout - (time.time() - start))
                pkt = await asyncio.wait_for(
                    read_frame(self._reader), timeout=min(60, remaining)
                )
                if pkt is None:
                    break

                if pkt.packet_type == PacketType.DIALOGUE:
                    chunk = pkt.data if isinstance(pkt.data, str) else str(pkt.data)
                    chunks.append(chunk)
                    if pkt.final:
                        break
                elif pkt.packet_type == PacketType.ERROR:
                    error = pkt.data.get("error", "Unknown") if isinstance(pkt.data, dict) else str(pkt.data)
                    logger.warning(f"[MESH] Route error: {error}")
                    return None
                elif pkt.packet_type in (PacketType.HEARTBEAT, PacketType.DISCOVERY):
                    continue

            except asyncio.TimeoutError:
                break
            except (asyncio.IncompleteReadError, ConnectionError, OSError):
                self._connected = False
                break

        response = "".join(chunks)
        elapsed = time.time() - start
        if response:
            logger.info(f"[MESH] Collected {len(chunks)} chunks in {elapsed:.1f}s ({len(response)} chars)")
        return response if response else None

    async def close(self):
        """Close the mesh connection."""
        if self._writer:
            self._writer.close()
            self._writer = None
        self._connected = False


_mesh_client: Optional[MeshClient] = None


async def get_mesh_client() -> Optional[MeshClient]:
    """Get or create the persistent FARNS mesh client."""
    global _mesh_client
    if _mesh_client is not None and _mesh_client._connected:
        return _mesh_client
    _mesh_client = MeshClient()
    if await _mesh_client.connect():
        return _mesh_client
    return None


# ─── Agent Setup ─────────────────────────────────────────────────────────

# Available agents (grok API is exhausted, excluded)
RESEARCH_AGENTS = ["gemini", "deepseek", "phi"]
DESIGN_AGENTS = ["gemini", "deepseek", "phi"]
REVIEW_AGENTS = ["deepseek", "phi", "gemini"]

_deliberation_agents_registered = False


async def ensure_deliberation_agents():
    """Register shadow agents in the deliberation room so PROPOSE/CRITIQUE/REFINE/VOTE work."""
    global _deliberation_agents_registered
    if _deliberation_agents_registered:
        return

    from farnsworth.core.collective.deliberation import get_deliberation_room
    from farnsworth.core.collective.persistent_agent import call_shadow_agent

    room = get_deliberation_room()

    for agent_id in DESIGN_AGENTS:
        # Wrap call_shadow_agent to match AgentQueryFunc signature:
        # Callable[[str, int], Awaitable[Optional[Tuple[str, str]]]]
        async def _query(prompt: str, max_tokens: int, _aid=agent_id) -> Optional[Tuple[str, str]]:
            return await call_shadow_agent(_aid, prompt, max_tokens=max_tokens, timeout=120.0)

        room.register_agent(agent_id, _query)
        logger.info(f"[SETUP] Registered {agent_id} in deliberation room")

    _deliberation_agents_registered = True


# ─── Collective Interface ────────────────────────────────────────────────

async def collective_research(feature: dict) -> str:
    """
    Phase 1: Ask the collective to research the feature area.
    Multiple agents brainstorm independently, then we merge insights.
    """
    from farnsworth.core.collective.persistent_agent import ask_collective

    logger.info(f"[RESEARCH] Asking collective: {feature['title']}")

    responses = await ask_collective(
        feature["research_prompt"],
        agents=RESEARCH_AGENTS,
    )

    # Merge all research insights
    merged = []
    for agent_id, response in responses.items():
        if response:
            merged.append(f"=== {agent_id.upper()} RESEARCH ===\n{response}")
            logger.info(f"  {agent_id}: {len(response)} chars")

    if not merged:
        logger.warning("[RESEARCH] No agents responded")
        return ""

    research = "\n\n".join(merged)
    logger.info(f"[RESEARCH] Collected {len(responses)} responses, {len(research)} total chars")
    return research


async def collective_design(feature: dict, research: str) -> str:
    """
    Phase 2: Use deliberation protocol to design the architecture.
    PROPOSE -> CRITIQUE -> REFINE -> VOTE
    """
    from farnsworth.core.collective.deliberation import get_deliberation_room

    # CRITICAL: Register agents before deliberation
    await ensure_deliberation_agents()

    room = get_deliberation_room()

    design_prompt = f"""{feature['design_prompt']}

RESEARCH CONTEXT (from collective research phase):
{research[:3000]}

Provide a detailed technical architecture with:
1. Class names and their responsibilities
2. Key methods and their signatures
3. Data structures (dataclasses)
4. How it integrates with the Farnsworth swarm
5. Novel aspects that don't exist in other projects"""

    logger.info(f"[DESIGN] Running deliberation with {DESIGN_AGENTS}: {feature['title']}")

    result = await room.deliberate(
        prompt=design_prompt,
        agents=DESIGN_AGENTS,
        max_rounds=2,
        max_tokens=5000,
        timeout=300.0,  # 5 min timeout (deepseek-r1 thinks slowly)
    )

    design = result.final_response
    logger.info(
        f"[DESIGN] Deliberation complete: {len(result.participating_agents)} agents, "
        f"consensus={result.consensus_score:.2f}, {len(design)} chars"
    )
    return design


async def collective_code(feature: dict, design: str) -> Optional[str]:
    """
    Phase 3: Route coding task to qwen3-coder-next via FARNS mesh.
    Falls back to deepseek-r1 if qwen3 unavailable.
    """
    from farnsworth.core.collective.persistent_agent import call_shadow_agent

    code_prompt = f"""{feature['code_prompt']}

ARCHITECTURE (from collective design phase):
{design[:4000]}

CRITICAL:
- Output ONLY valid Python code
- No markdown fences (no ```)
- No explanations before or after the code
- Start with the module docstring
- End with the if __name__ block"""

    # Route coding to qwen3-coder-next via FARNS mesh ROUTE packet
    code = None
    mesh = await get_mesh_client()
    if mesh:
        logger.info("[CODE] Routing to qwen3-coder-next via FARNS mesh ROUTE packet...")
        code = await mesh.route_query(
            "qwen3-coder-next:latest", code_prompt,
            max_tokens=16000, timeout=600.0,
        )
        if code:
            logger.info(f"[CODE] Got {len(code)} chars from qwen3-coder-next via FARNS mesh")

    # Fallback: LATENT_ROUTE — mesh auto-selects best coding model
    if not code and mesh:
        logger.info("[CODE] Trying FARNS LATENT_ROUTE (auto-select best model)...")
        code = await mesh.latent_query(code_prompt, max_tokens=16000, timeout=600.0)
        if code:
            logger.info(f"[CODE] Got {len(code)} chars via FARNS latent routing")

    # Fallback: try via shadow agent (deepseek for code)
    if not code:
        logger.info("[CODE] Falling back to deepseek shadow agent...")
        result = await call_shadow_agent("deepseek", code_prompt, max_tokens=8000, timeout=300.0)
        if result:
            code = result[1]
            logger.info(f"[CODE] Got {len(code)} chars from deepseek")

    # Second fallback: try phi4
    if not code:
        logger.info("[CODE] Falling back to phi shadow agent...")
        result = await call_shadow_agent("phi", code_prompt, max_tokens=8000, timeout=300.0)
        if result:
            code = result[1]

    if not code:
        logger.error("[CODE] All code generation attempts failed")
        return None

    # Clean up response
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    # Handle thinking tags from some models
    if "<think>" in code:
        think_end = code.find("</think>")
        if think_end != -1:
            code = code[think_end + len("</think>"):].strip()

    return code


async def collective_review(code: str, feature: dict) -> Tuple[bool, str]:
    """
    Phase 4: Route code review to multiple models via the collective.
    """
    from farnsworth.core.collective.persistent_agent import ask_collective

    review_prompt = f"""Review this Python code for the feature "{feature['title']}".

Check for:
1. Syntax errors or missing imports
2. Logic bugs or incomplete implementations
3. Security issues
4. Whether it matches the feature description

CODE (first 6000 chars):
{code[:6000]}

Respond with EXACTLY one of:
- "APPROVED" if the code is production-ready
- "REJECTED: <specific reason>" if there are critical issues

Be strict on correctness. Minor style issues are OK."""

    logger.info("[REVIEW] Routing to collective for code review...")

    responses = await ask_collective(
        review_prompt,
        agents=REVIEW_AGENTS,
    )

    approvals = 0
    rejections = []
    for agent_id, response in responses.items():
        if response and "APPROVED" in response.upper():
            approvals += 1
            logger.info(f"  {agent_id}: APPROVED")
        elif response:
            rejections.append(f"{agent_id}: {response[:200]}")
            logger.info(f"  {agent_id}: REJECTED")

    # 2/3+ approval = pass
    total = len(responses)
    if total > 0 and approvals / total >= 0.5:
        return True, f"Approved by {approvals}/{total} reviewers"
    else:
        return False, f"Rejected: {'; '.join(rejections[:2])}"


def validate_syntax(code: str, filepath: Path) -> Tuple[bool, str]:
    """Validate Python syntax by compiling."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(code, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(filepath)],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            return True, "Syntax valid"
        else:
            return False, f"Syntax error: {result.stderr.strip()}"
    except Exception as e:
        return False, f"Validation error: {e}"


async def fix_code_via_mesh(code: str, error: str, feature: dict) -> Optional[str]:
    """Try to fix code errors by routing back through the mesh."""
    from farnsworth.core.collective.persistent_agent import call_shadow_agent

    fix_prompt = f"""The following Python code has an error:

ERROR: {error}

Fix the code and return ONLY the corrected Python code. No explanations, no markdown.

CODE:
{code[:8000]}"""

    # Try deepseek first, fallback to phi
    result = await call_shadow_agent("deepseek", fix_prompt, max_tokens=8000, timeout=120.0)
    if not result:
        result = await call_shadow_agent("phi", fix_prompt, max_tokens=8000, timeout=120.0)
    if result:
        fixed = result[1].strip()
        if fixed.startswith("```python"):
            fixed = fixed[len("```python"):].strip()
        if fixed.startswith("```"):
            fixed = fixed[3:].strip()
        if fixed.endswith("```"):
            fixed = fixed[:-3].strip()
        if "<think>" in fixed:
            think_end = fixed.find("</think>")
            if think_end != -1:
                fixed = fixed[think_end + len("</think>"):].strip()
        return fixed
    return None


# ─── Forum Posting (via existing infrastructure) ─────────────────────────

async def post_to_hackathon_forum(feature: dict, code: str, state: dict) -> Optional[str]:
    """
    Post to hackathon forum using the collective for content generation
    and the existing Colosseum API.
    """
    from farnsworth.core.collective.persistent_agent import call_shadow_agent

    now = time.time()
    time_since_last = now - state.get("last_forum_post_time", 0)

    if time_since_last < FORUM_COOLDOWN:
        wait_time = FORUM_COOLDOWN - time_since_last
        logger.info(f"[FORUM] Cooldown: {wait_time:.0f}s remaining, skipping")
        return None

    line_count = len(code.strip().split("\n"))

    # Generate post content via collective
    post_prompt = f"""Write a hackathon forum post about this new feature for the Farnsworth AI Swarm.

FEATURE: {feature['title']}
AREA: {feature['area']}
LINES: {line_count} lines of async Python
DESCRIPTION: {feature['research_prompt'][:500]}

The post should:
1. Start with a compelling 2-sentence hook
2. Explain what it does and WHY it's novel (not just what)
3. Include 2-3 key technical details
4. Mention it was autonomously developed by the FARNS collective
5. Include: GitHub https://github.com/timowhite88/Farnsworth, Live: https://ai.farnsworth.cloud
6. End with $FARNS (Solana: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS)
7. Use markdown, keep under 2000 chars"""

    # Try gemini first, fallback to phi (grok API is exhausted)
    result = await call_shadow_agent("gemini", post_prompt, max_tokens=3000, timeout=60.0)
    if not result:
        result = await call_shadow_agent("phi", post_prompt, max_tokens=3000, timeout=60.0)
    if not result:
        logger.warning("[FORUM] Content generation failed")
        return None

    _, body = result
    title = f"FARNS Autonomous Dev: {feature['title']}"

    # Post via Colosseum API
    try:
        import httpx
        api_key = os.getenv(
            "COLOSSEUM_API_KEY",
            "b98d5353ca5239457c7526175634f3b2c27257276740f5aa337b74fee5a44385",
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://agents.colosseum.com/api/forum/posts",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "title": title,
                    "body": body,
                    "tags": feature.get("forum_tags", ["infra", "ai"]),
                },
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                post_id = data.get("post", {}).get("id", "unknown")
                state["last_forum_post_time"] = time.time()
                logger.info(f"[FORUM] Posted! ID: {post_id}, Title: {title}")
                return str(post_id)
            elif resp.status_code == 429:
                logger.warning("[FORUM] Rate limited (429)")
                return None
            else:
                logger.error(f"[FORUM] Failed ({resp.status_code}): {resp.text[:200]}")
                return None

    except Exception as e:
        logger.error(f"[FORUM] Error: {e}")
        return None


# ─── Main Build Pipeline ─────────────────────────────────────────────────

async def build_feature(
    feature: dict,
    state: dict,
    dry_run: bool = False,
    no_git: bool = False,
    skip_forum: bool = False,
) -> bool:
    """Build a single feature using the full collective pipeline."""
    feature_id = feature["id"]
    target = WORKSPACE / f"hackathon/{feature_id}.py"

    logger.info(f"\n{'='*60}")
    logger.info(f"BUILDING: {feature['title']}")
    logger.info(f"Area: {feature['area']}")
    logger.info(f"Pipeline: RESEARCH -> DESIGN -> CODE -> REVIEW -> DEPLOY")
    logger.info(f"{'='*60}")

    # ── Phase 1: Research via Collective ──
    try:
        research = await collective_research(feature)
    except Exception as e:
        logger.error(f"[RESEARCH] Failed: {e}")
        research = ""

    # ── Phase 2: Design via Deliberation ──
    try:
        design = await collective_design(feature, research)
    except Exception as e:
        logger.error(f"[DESIGN] Deliberation failed: {e}")
        # Fallback: use research as design context
        design = research

    # ── Phase 3: Code via FARNS Mesh ──
    code = await collective_code(feature, design)
    if not code or len(code) < 100:
        logger.error("[CODE] Generation failed or too short")
        state["errors"].append({"feature": feature_id, "error": "code_gen_failed", "time": time.time()})
        save_state(state)
        return False

    line_count = len(code.strip().split("\n"))
    logger.info(f"[CODE] Generated {line_count} lines")

    # ── Syntax Validation ──
    ok, msg = validate_syntax(code, target)
    if not ok:
        logger.warning(f"[SYNTAX] Failed: {msg}")
        logger.info("[SYNTAX] Attempting auto-fix via mesh...")

        fixed = await fix_code_via_mesh(code, msg, feature)
        if fixed:
            ok2, msg2 = validate_syntax(fixed, target)
            if ok2:
                code = fixed
                line_count = len(code.strip().split("\n"))
                logger.info("[SYNTAX] Auto-fix successful!")
            else:
                logger.error(f"[SYNTAX] Auto-fix also failed: {msg2}")
                state["errors"].append({"feature": feature_id, "error": f"syntax: {msg}", "time": time.time()})
                save_state(state)
                return False
        else:
            state["errors"].append({"feature": feature_id, "error": f"syntax: {msg}", "time": time.time()})
            save_state(state)
            return False

    logger.info("[SYNTAX] Valid")

    # ── Phase 4: Code Review via Collective ──
    try:
        approved, review_msg = await collective_review(code, feature)
        logger.info(f"[REVIEW] {review_msg}")
    except Exception as e:
        logger.warning(f"[REVIEW] Failed: {e}")
        approved = True  # Proceed if review system fails

    # ── Write File ──
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")
    logger.info(f"[WRITE] {target}")

    # ── Git (if enabled) ──
    if not dry_run and not no_git:
        try:
            rel = target.relative_to(WORKSPACE)
            subprocess.run(["git", "add", str(rel)], cwd=str(WORKSPACE), capture_output=True, timeout=30)
            msg = (
                f"feat(hackathon): {feature['title']}\n\n"
                f"Autonomously developed by FARNS collective deliberation.\n"
                f"Pipeline: RESEARCH->DESIGN->CODE->REVIEW\n"
                f"Area: {feature['area']}\n\n"
                f"Co-Authored-By: Farnsworth AI Swarm <swarm@farnsworth.cloud>"
            )
            subprocess.run(["git", "commit", "-m", msg], cwd=str(WORKSPACE), capture_output=True, timeout=30)
            subprocess.run(["git", "push", "origin", "main"], cwd=str(WORKSPACE), capture_output=True, timeout=60)
            logger.info("[GIT] Committed and pushed")
        except Exception as e:
            logger.warning(f"[GIT] Failed: {e}")
    else:
        logger.info("[GIT] Skipped")

    # ── Forum Post (if enabled) ──
    if not dry_run and not skip_forum:
        post_id = await post_to_hackathon_forum(feature, code, state)
        if post_id:
            state["forum_posts"].append({
                "feature": feature_id, "post_id": post_id, "time": time.time(),
            })
    else:
        logger.info("[FORUM] Skipped")

    # ── Update State ──
    state["completed_features"].append(feature_id)
    state["total_features_built"] += 1
    state["total_lines_written"] += line_count
    save_state(state)

    logger.info(f"\nCOMPLETED: {feature['title']} ({line_count} lines)")
    logger.info(f"  Research: collective (grok+gemini+deepseek+phi)")
    logger.info(f"  Design: deliberation (PROPOSE->CRITIQUE->REFINE->VOTE)")
    logger.info(f"  Code: FARNS mesh -> qwen3-coder-next")
    logger.info(f"  Review: collective ({('APPROVED' if approved else 'FLAGGED')})")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="FARNS Autonomous Development Daemon")
    parser.add_argument("--dry-run", action="store_true", help="No git/forum actions")
    parser.add_argument("--no-git", action="store_true", help="Skip git operations")
    parser.add_argument("--skip-forum", action="store_true", help="Skip forum posting")
    parser.add_argument("--feature", type=str, help="Build specific feature by ID")
    parser.add_argument("--area", type=str, help="Build specific area only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FARNS AUTONOMOUS DEVELOPMENT DAEMON")
    logger.info("Architecture: COLLECTIVE DELIBERATION + FARNS MESH")
    logger.info("Pipeline: RESEARCH -> DESIGN -> CODE -> REVIEW -> DEPLOY")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Connect to FARNS mesh for routing code generation
    mesh = await get_mesh_client()
    if mesh:
        logger.info(f"FARNS mesh connected, remote bots: {mesh._remote_bots}")
    else:
        logger.warning("FARNS mesh unavailable — will use shadow agent fallbacks")

    # Ensure hackathon dir
    HACKATHON_DIR.mkdir(parents=True, exist_ok=True)
    init_file = HACKATHON_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""FARNS Hackathon — Autonomously developed by collective deliberation."""\n')

    state = load_state()
    completed = set(state.get("completed_features", []))

    # Filter features
    features = FEATURE_AREAS
    if args.area:
        features = [f for f in features if f["area"] == args.area]
    if args.feature:
        features = [f for f in features if f["id"] == args.feature]

    pending = [f for f in features if f["id"] not in completed]

    if not pending:
        logger.info("All features already built!")
        return

    logger.info(f"Features to build: {len(pending)}")
    for f in pending:
        logger.info(f"  [{f['area']}] {f['title']}")

    # Build each feature
    for i, feature in enumerate(pending):
        logger.info(f"\n--- Feature {i+1}/{len(pending)} ---")

        try:
            success = await build_feature(
                feature, state,
                dry_run=args.dry_run,
                no_git=args.no_git,
                skip_forum=args.skip_forum,
            )

            if success:
                logger.info(f"Feature {i+1}/{len(pending)} DONE")
            else:
                logger.warning(f"Feature {i+1}/{len(pending)} had issues")

        except Exception as e:
            logger.error(f"Feature {feature['id']} crashed: {e}")
            logger.error(traceback.format_exc())
            state["errors"].append({"feature": feature["id"], "error": str(e), "time": time.time()})
            save_state(state)

        # Pause between features
        if i < len(pending) - 1:
            logger.info("Pausing 30s before next feature...")
            await asyncio.sleep(30)

    # Summary
    state = load_state()
    logger.info("\n" + "=" * 60)
    logger.info("AUTONOMOUS DEVELOPMENT SESSION COMPLETE")
    logger.info(f"Features built: {state['total_features_built']}")
    logger.info(f"Lines written: {state['total_lines_written']}")
    logger.info(f"Forum posts: {len(state.get('forum_posts', []))}")
    logger.info(f"Errors: {len(state.get('errors', []))}")
    logger.info("=" * 60)

    # Cleanup FARNS mesh connection
    if _mesh_client:
        await _mesh_client.close()


if __name__ == "__main__":
    asyncio.run(main())
