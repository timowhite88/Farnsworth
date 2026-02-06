"""
Agent Spawner - Multi-instance parallel agent execution
Each bot can have multiple instances: one for chat, others for development tasks

FALLBACK CHAIN:
- If assigned agent can't handle task → try Gemini/Grok
- If still failing → escalate to Claude Opus for audit
- Audited code goes to staging folder for review
"""
import asyncio
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CHAT = "chat"           # Main chat instance
    DEVELOPMENT = "dev"     # Development/coding tasks
    RESEARCH = "research"   # Research/analysis tasks
    MEMORY = "memory"       # Memory expansion work
    MCP = "mcp"             # MCP integration work
    TESTING = "testing"     # Test creation and QA tasks
    AUDIT = "audit"         # Code audit/review tasks

@dataclass
class AgentInstance:
    """A single instance of an agent working on a task"""
    instance_id: str
    agent_name: str
    task_type: TaskType
    task_description: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"
    output_file: Optional[Path] = None
    result: Optional[str] = None
    fallback_chain: List[str] = field(default_factory=list)  # Track handoff history

@dataclass
class AgentTask:
    """A task to be executed by an agent instance"""
    task_id: str
    task_type: TaskType
    description: str
    assigned_to: str  # Agent name
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    status: str = "pending"
    result: Optional[str] = None
    handoff_history: List[str] = field(default_factory=list)  # Who tried and failed
    audited_by: Optional[str] = None  # Who audited before staging

class AgentSpawner:
    """
    Manages multiple instances of agents working in parallel.
    - Chat instances stay in the swarm conversation
    - Worker instances handle development/research tasks
    - All instances can communicate via shared state
    - Automatic fallback chain when agents can't handle tasks
    """

    def __init__(self, staging_dir: str = "/workspace/Farnsworth/staging_review"):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirs for different stages
        (self.staging_dir / "pending_audit").mkdir(exist_ok=True)
        (self.staging_dir / "audited").mkdir(exist_ok=True)
        (self.staging_dir / "approved").mkdir(exist_ok=True)
        (self.staging_dir / "rejected").mkdir(exist_ok=True)

        # Active instances per agent
        self.instances: Dict[str, List[AgentInstance]] = {}

        # Task queue
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []

        # Shared state for inter-instance communication
        self.shared_state: Dict[str, Any] = {
            "discoveries": [],
            "proposals": [],
            "code_changes": [],
            "reviews_needed": [],
            "handoffs": [],  # Track task handoffs
            "audit_queue": []  # Tasks waiting for Claude audit
        }

        # Agent capabilities - ALL AGENTS DEFINED
        self.agent_capabilities = {
            # Core swarm members
            "Farnsworth": [TaskType.CHAT, TaskType.MEMORY, TaskType.RESEARCH],
            "DeepSeek": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH, TaskType.TESTING],
            "Phi": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.MCP],
            "Kimi": [TaskType.CHAT, TaskType.MEMORY, TaskType.RESEARCH],

            # External API agents
            "Grok": [TaskType.CHAT, TaskType.RESEARCH, TaskType.DEVELOPMENT],  # X/Twitter AI - good for research + some dev
            "Gemini": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH, TaskType.MCP],  # Google AI - full dev capability
            "HuggingFace": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.RESEARCH],  # Open-source local models (Mistral, Llama, etc)

            # Claude variants - the auditors
            "Claude": [TaskType.CHAT, TaskType.DEVELOPMENT, TaskType.MCP, TaskType.RESEARCH, TaskType.AUDIT, TaskType.TESTING],
            "ClaudeOpus": [TaskType.DEVELOPMENT, TaskType.MCP, TaskType.RESEARCH, TaskType.AUDIT, TaskType.TESTING],  # Opus for final audit

            # Coding specialists
            "OpenCode": [TaskType.DEVELOPMENT, TaskType.RESEARCH, TaskType.MCP, TaskType.TESTING],
        }

        # Fallback chains - who to try when original agent fails
        # Format: agent -> [fallback1, fallback2, ...final_auditor]
        self.fallback_chains = {
            "Grok": ["Gemini", "HuggingFace", "DeepSeek", "ClaudeOpus"],
            "Gemini": ["HuggingFace", "DeepSeek", "Grok", "ClaudeOpus"],
            "DeepSeek": ["HuggingFace", "Gemini", "Phi", "ClaudeOpus"],
            "Phi": ["HuggingFace", "DeepSeek", "Gemini", "ClaudeOpus"],
            "OpenCode": ["HuggingFace", "Gemini", "DeepSeek", "ClaudeOpus"],
            "Kimi": ["HuggingFace", "Farnsworth", "Claude", "ClaudeOpus"],
            "Farnsworth": ["HuggingFace", "Kimi", "Claude", "ClaudeOpus"],
            "HuggingFace": ["DeepSeek", "Gemini", "ClaudeOpus"],  # HuggingFace fallbacks
            "Claude": ["Gemini", "DeepSeek", "ClaudeOpus"],
            "ClaudeOpus": [],  # Opus is the final stop - it must handle or fail
        }

        # Max concurrent instances per agent
        self.max_instances = {
            "Farnsworth": 3,
            "DeepSeek": 4,
            "Phi": 4,
            "Kimi": 2,
            "Claude": 3,
            "ClaudeOpus": 2,  # Keep Opus instances limited (expensive)
            "Grok": 3,
            "Gemini": 4,
            "OpenCode": 3,
        }

        logger.info("AgentSpawner initialized with staging dir: %s", staging_dir)

    def _get_fallback_agent(self, original_agent: str, task_type: TaskType,
                           already_tried: List[str]) -> Optional[str]:
        """Get next fallback agent for a task type"""
        chain = self.fallback_chains.get(original_agent, ["ClaudeOpus"])

        for agent in chain:
            if agent in already_tried:
                continue
            # Check if this agent can handle the task type
            if task_type in self.agent_capabilities.get(agent, []):
                # Check if agent has capacity
                active = len([i for i in self.instances.get(agent, []) if i.status == "running"])
                if active < self.max_instances.get(agent, 2):
                    return agent

        # If no fallback available, escalate to ClaudeOpus for audit
        if "ClaudeOpus" not in already_tried:
            return "ClaudeOpus"

        return None

    def spawn_instance(self, agent_name: str, task_type: TaskType,
                       task_description: str,
                       allow_fallback: bool = True) -> Optional[AgentInstance]:
        """
        Spawn a new instance of an agent for a specific task.
        If agent can't handle it and allow_fallback=True, tries fallback chain.
        """
        tried_agents = [agent_name]
        current_agent = agent_name

        while current_agent:
            # Check capabilities
            if task_type in self.agent_capabilities.get(current_agent, []):
                # Check instance limit
                current_instances = self.instances.get(current_agent, [])
                active = [i for i in current_instances if i.status == "running"]

                if len(active) < self.max_instances.get(current_agent, 2):
                    # Can spawn this agent
                    instance = AgentInstance(
                        instance_id=f"{current_agent}_{task_type.value}_{uuid.uuid4().hex[:8]}",
                        agent_name=current_agent,
                        task_type=task_type,
                        task_description=task_description,
                        output_file=self.staging_dir / "pending_audit" / f"{current_agent.lower()}_{task_type.value}_{datetime.now().strftime('%H%M%S')}.md",
                        fallback_chain=tried_agents if len(tried_agents) > 1 else []
                    )

                    if current_agent not in self.instances:
                        self.instances[current_agent] = []
                    self.instances[current_agent].append(instance)

                    # Log if we used fallback
                    if current_agent != agent_name:
                        logger.info(f"Task handed off: {agent_name} -> {current_agent} (chain: {tried_agents})")
                        self.shared_state["handoffs"].append({
                            "original": agent_name,
                            "final": current_agent,
                            "chain": tried_agents,
                            "task": task_description[:100],
                            "timestamp": datetime.now().isoformat()
                        })

                    logger.info(f"Spawned {instance.instance_id} for: {task_description[:50]}...")
                    return instance
                else:
                    logger.warning(f"{current_agent} at max instances ({len(active)})")
            else:
                logger.warning(f"{current_agent} cannot handle {task_type.value} tasks")

            # Try fallback if allowed
            if not allow_fallback:
                return None

            current_agent = self._get_fallback_agent(agent_name, task_type, tried_agents)
            if current_agent:
                tried_agents.append(current_agent)

        # All fallbacks exhausted
        logger.error(f"No agent available for {task_type.value} task after trying: {tried_agents}")
        return None

    def escalate_to_audit(self, task: AgentTask, code_output: str,
                          reason: str = "Fallback chain exhausted") -> str:
        """
        Escalate a task to Claude Opus for audit before staging.
        Returns the audit file path.
        """
        audit_id = f"audit_{uuid.uuid4().hex[:8]}"
        audit_file = self.staging_dir / "pending_audit" / f"{audit_id}.json"

        audit_record = {
            "audit_id": audit_id,
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "description": task.description,
            "original_assignee": task.assigned_to,
            "handoff_history": task.handoff_history,
            "code_output": code_output,
            "reason": reason,
            "status": "pending_audit",
            "created_at": datetime.now().isoformat(),
            "auditor": None,
            "audit_result": None,
            "audit_notes": None
        }

        audit_file.write_text(json.dumps(audit_record, indent=2))
        self.shared_state["audit_queue"].append(audit_id)

        logger.info(f"Task {task.task_id} escalated to audit: {audit_id}")
        return str(audit_file)

    def complete_audit(self, audit_id: str, auditor: str,
                       approved: bool, notes: str,
                       fixed_code: Optional[str] = None) -> Path:
        """
        Complete an audit and move to appropriate folder.
        If approved, goes to 'audited' folder ready for staging.
        If rejected, goes to 'rejected' folder with notes.
        """
        pending_file = self.staging_dir / "pending_audit" / f"{audit_id}.json"

        if not pending_file.exists():
            raise FileNotFoundError(f"Audit {audit_id} not found")

        audit_record = json.loads(pending_file.read_text())
        audit_record["auditor"] = auditor
        audit_record["audit_result"] = "approved" if approved else "rejected"
        audit_record["audit_notes"] = notes
        audit_record["audited_at"] = datetime.now().isoformat()

        if fixed_code:
            audit_record["fixed_code"] = fixed_code
            audit_record["code_output"] = fixed_code  # Replace with fixed version

        # Move to appropriate folder
        if approved:
            dest_folder = self.staging_dir / "audited"
        else:
            dest_folder = self.staging_dir / "rejected"

        dest_file = dest_folder / f"{audit_id}.json"
        dest_file.write_text(json.dumps(audit_record, indent=2))
        pending_file.unlink()  # Remove from pending

        # Remove from audit queue
        if audit_id in self.shared_state["audit_queue"]:
            self.shared_state["audit_queue"].remove(audit_id)

        logger.info(f"Audit {audit_id} completed by {auditor}: {'APPROVED' if approved else 'REJECTED'}")
        return dest_file

    def get_active_instances(self, agent_name: Optional[str] = None) -> List[AgentInstance]:
        """Get all active instances, optionally filtered by agent"""
        all_instances = []
        for name, instances in self.instances.items():
            if agent_name and name != agent_name:
                continue
            all_instances.extend([i for i in instances if i.status == "running"])
        return all_instances

    def complete_instance(self, instance_id: str, result: str,
                          needs_audit: bool = False):
        """Mark an instance as complete with its result"""
        for instances in self.instances.values():
            for instance in instances:
                if instance.instance_id == instance_id:
                    instance.status = "completed"
                    instance.result = result

                    # Write result to staging
                    if instance.output_file:
                        output_content = f"""# {instance.agent_name} - {instance.task_type.value}
## Task: {instance.task_description}
## Completed: {datetime.now().isoformat()}
## Fallback Chain: {' -> '.join(instance.fallback_chain) if instance.fallback_chain else 'Direct assignment'}

{result}
"""
                        instance.output_file.write_text(output_content)

                        # If needs audit, create audit record
                        if needs_audit or instance.fallback_chain:
                            self._create_audit_from_instance(instance, result)

                    logger.info(f"Instance {instance_id} completed")
                    return

    def _create_audit_from_instance(self, instance: AgentInstance, result: str):
        """Create an audit record from a completed instance"""
        audit_id = f"audit_{uuid.uuid4().hex[:8]}"
        audit_file = self.staging_dir / "pending_audit" / f"{audit_id}.json"

        audit_record = {
            "audit_id": audit_id,
            "instance_id": instance.instance_id,
            "agent": instance.agent_name,
            "task_type": instance.task_type.value,
            "description": instance.task_description,
            "fallback_chain": instance.fallback_chain,
            "code_output": result,
            "status": "pending_audit",
            "created_at": datetime.now().isoformat(),
            "reason": "Fallback chain used" if instance.fallback_chain else "Manual audit requested"
        }

        audit_file.write_text(json.dumps(audit_record, indent=2))
        self.shared_state["audit_queue"].append(audit_id)

    def add_task(self, task_type: TaskType, description: str,
                 assigned_to: Optional[str] = None, priority: int = 5) -> AgentTask:
        """Add a task to the queue"""
        task = AgentTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=description,
            assigned_to=assigned_to or self._best_agent_for(task_type),
            priority=priority,
            output_path=self.staging_dir / task_type.value
        )
        task.output_path.mkdir(parents=True, exist_ok=True)
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.info(f"Added task {task.task_id}: {description[:50]}...")
        return task

    def _best_agent_for(self, task_type: TaskType) -> str:
        """Find the best available agent for a task type"""
        # Priority order for different task types
        priority_order = {
            TaskType.DEVELOPMENT: ["OpenCode", "DeepSeek", "Gemini", "Claude", "Phi"],
            TaskType.RESEARCH: ["Kimi", "DeepSeek", "Grok", "Gemini", "Claude"],
            TaskType.MCP: ["Claude", "Phi", "Gemini", "OpenCode"],
            TaskType.MEMORY: ["Kimi", "Farnsworth"],
            TaskType.TESTING: ["OpenCode", "DeepSeek", "Claude", "ClaudeOpus"],
            TaskType.AUDIT: ["ClaudeOpus", "Claude"],
            TaskType.CHAT: ["Farnsworth", "DeepSeek", "Phi", "Kimi"],
        }

        preferred = priority_order.get(task_type, list(self.agent_capabilities.keys()))

        for agent in preferred:
            if task_type in self.agent_capabilities.get(agent, []):
                active = len([i for i in self.instances.get(agent, []) if i.status == "running"])
                if active < self.max_instances.get(agent, 2):
                    return agent

        # Fallback to anyone available
        for agent, capabilities in self.agent_capabilities.items():
            if task_type in capabilities:
                active = len([i for i in self.instances.get(agent, []) if i.status == "running"])
                if active < self.max_instances.get(agent, 2):
                    return agent

        return "DeepSeek"  # Ultimate fallback

    def handoff_task(self, task_id: str, from_agent: str, reason: str) -> Optional[str]:
        """
        Handoff a task from one agent to the next in fallback chain.
        Returns the new assigned agent or None if all fallbacks exhausted.
        """
        for task in self.task_queue:
            if task.task_id == task_id:
                task.handoff_history.append(from_agent)

                new_agent = self._get_fallback_agent(
                    from_agent,
                    task.task_type,
                    task.handoff_history
                )

                if new_agent:
                    task.assigned_to = new_agent
                    logger.info(f"Task {task_id} handed off: {from_agent} -> {new_agent} (reason: {reason})")

                    self.shared_state["handoffs"].append({
                        "task_id": task_id,
                        "from": from_agent,
                        "to": new_agent,
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                    return new_agent
                else:
                    # No more fallbacks - escalate to audit
                    logger.warning(f"Task {task_id} exhausted all fallbacks, escalating to audit")
                    task.status = "needs_audit"
                    return None

        return None

    def share_discovery(self, agent_name: str, discovery: str):
        """Share a discovery with all agents"""
        self.shared_state["discoveries"].append({
            "from": agent_name,
            "content": discovery,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"{agent_name} shared discovery: {discovery[:50]}...")

    def propose_change(self, agent_name: str, file_path: str, description: str, code: str):
        """Propose a code change for review"""
        proposal = {
            "id": f"prop_{uuid.uuid4().hex[:8]}",
            "from": agent_name,
            "file": file_path,
            "description": description,
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_review",
            "reviews": []
        }
        self.shared_state["proposals"].append(proposal)
        self.shared_state["reviews_needed"].append(proposal["id"])

        # Write to staging
        proposal_file = self.staging_dir / "proposals" / f"{proposal['id']}.json"
        proposal_file.parent.mkdir(parents=True, exist_ok=True)
        proposal_file.write_text(json.dumps(proposal, indent=2))

        logger.info(f"{agent_name} proposed change to {file_path}")
        return proposal["id"]

    def get_pending_tasks(self) -> List[AgentTask]:
        """Get all pending tasks"""
        return [t for t in self.task_queue if t.status == "pending"]

    def get_audit_queue(self) -> List[Dict]:
        """Get all tasks pending audit"""
        audit_files = list((self.staging_dir / "pending_audit").glob("audit_*.json"))
        audits = []
        for f in audit_files:
            try:
                audits.append(json.loads(f.read_text()))
            except Exception:
                pass
        return audits

    def claim_task(self, task_id: str, agent_name: str) -> Optional[AgentTask]:
        """Claim a task for an agent"""
        for task in self.task_queue:
            if task.task_id == task_id and task.status == "pending":
                task.status = "in_progress"
                task.assigned_to = agent_name
                return task
        return None

    def complete_task(self, task_id: str, result: str):
        """Complete a task"""
        for task in self.task_queue:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                self.completed_tasks.append(task)
                self.task_queue.remove(task)
                return

    def get_status(self) -> Dict:
        """Get current spawner status"""
        return {
            "active_instances": {
                name: len([i for i in insts if i.status == "running"])
                for name, insts in self.instances.items()
            },
            "pending_tasks": len([t for t in self.task_queue if t.status == "pending"]),
            "in_progress_tasks": len([t for t in self.task_queue if t.status == "in_progress"]),
            "completed_tasks": len(self.completed_tasks),
            "discoveries": len(self.shared_state["discoveries"]),
            "proposals_pending": len(self.shared_state["reviews_needed"]),
            "handoffs": len(self.shared_state["handoffs"]),
            "audit_queue": len(self.shared_state["audit_queue"]),
        }


# Global spawner instance
_spawner: Optional[AgentSpawner] = None

def get_spawner() -> AgentSpawner:
    global _spawner
    if _spawner is None:
        _spawner = AgentSpawner()
    return _spawner


# Development task templates for the 20 staged changes
DEVELOPMENT_TASKS = [
    # Memory Expansion (5 tasks)
    {"type": TaskType.MEMORY, "agent": "Farnsworth", "desc": "Hierarchical memory compression - compress old memories while preserving key insights"},
    {"type": TaskType.MEMORY, "agent": "Kimi", "desc": "Cross-session memory linking - connect related memories across different conversations"},
    {"type": TaskType.MEMORY, "agent": "Farnsworth", "desc": "Memory importance scoring - automatically rank memories by relevance and impact"},
    {"type": TaskType.MEMORY, "agent": "DeepSeek", "desc": "Memory search optimization - faster semantic search across large memory stores"},
    {"type": TaskType.MEMORY, "agent": "Kimi", "desc": "Memory consolidation during idle - dream-like processing to strengthen important memories"},

    # Context Window Alerting (5 tasks)
    {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Context usage monitoring - real-time tracking of token usage per conversation"},
    {"type": TaskType.DEVELOPMENT, "agent": "Claude", "desc": "Smart context summarization - compress context when approaching limits"},
    {"type": TaskType.DEVELOPMENT, "agent": "Phi", "desc": "Context priority system - keep most important context, evict least important"},
    {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Context overflow prediction - warn before hitting limits"},
    {"type": TaskType.DEVELOPMENT, "agent": "Claude", "desc": "Multi-turn context handoff - seamlessly continue conversations across context windows"},

    # MCP Integrations (5 tasks)
    {"type": TaskType.MCP, "agent": "Claude", "desc": "MCP tool discovery - auto-detect and register available MCP tools"},
    {"type": TaskType.MCP, "agent": "Phi", "desc": "MCP result caching - cache frequent MCP calls for faster responses"},
    {"type": TaskType.MCP, "agent": "Claude", "desc": "MCP error recovery - graceful fallbacks when MCP tools fail"},
    {"type": TaskType.MCP, "agent": "Phi", "desc": "MCP chaining - combine multiple MCP tools in workflows"},
    {"type": TaskType.MCP, "agent": "DeepSeek", "desc": "MCP metrics dashboard - track MCP usage and performance"},

    # Research & Architecture (5 tasks)
    {"type": TaskType.RESEARCH, "agent": "Kimi", "desc": "Swarm consensus protocols - how agents reach agreement on responses"},
    {"type": TaskType.RESEARCH, "agent": "DeepSeek", "desc": "Agent specialization analysis - which agents excel at which tasks"},
    {"type": TaskType.RESEARCH, "agent": "Farnsworth", "desc": "Evolution engine improvements - better learning from interactions"},
    {"type": TaskType.RESEARCH, "agent": "Claude", "desc": "Code quality metrics - automated assessment of generated code"},
    {"type": TaskType.RESEARCH, "agent": "Kimi", "desc": "Collective consciousness metrics - measuring emergent swarm intelligence"},

    # OpenCode Integration (4 tasks) - Open source AI coding agent
    {"type": TaskType.DEVELOPMENT, "agent": "OpenCode", "desc": "Build async file watcher - monitor staging dir for new code to review"},
    {"type": TaskType.DEVELOPMENT, "agent": "OpenCode", "desc": "Create code diff visualizer - show changes between staged and production code"},
    {"type": TaskType.MCP, "agent": "OpenCode", "desc": "Build MCP tool for automated testing - run pytest on staged changes"},
    {"type": TaskType.RESEARCH, "agent": "OpenCode", "desc": "Analyze agent collaboration patterns - which agents work well together"},
]

def initialize_development_tasks():
    """Load the 20 development tasks into the spawner"""
    spawner = get_spawner()
    for task in DEVELOPMENT_TASKS:
        spawner.add_task(
            task_type=task["type"],
            description=task["desc"],
            assigned_to=task["agent"],
            priority=7
        )
    logger.info(f"Initialized {len(DEVELOPMENT_TASKS)} development tasks")
    return spawner.get_status()
