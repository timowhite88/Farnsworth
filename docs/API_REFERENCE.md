# Farnsworth API Reference

## Core Concepts

Farnsworth is designed around three main pillars:
1. **Memory**: Persistent storage for context, facts, and relationships.
2. **Agents**: Specialist workers for code, reasoning, and research.
3. **Evolution**: Self-improving feedback loops.

## Python SDK

The `FarnsworthClient` is the primary entry point for Python integrations.

```python
from farnsworth.client import FarnsworthClient

client = FarnsworthClient()
client.remember("Project X deadline is Friday")
```

### Methods

#### `remember(content: str, tags: list[str] = None)`
Stores information in the semantic and episodic memory layers.

#### `recall(query: str, limit: int = 5)`
Retrieves relevant memories based on semantic similarity.

#### `delegate_task(task: str)`
Dispatches a task to the Agent Swarm. The system automatically selects the best agent (Planner, Coder, etc.).

## MCP Protocol

For integration with Claude Code or other MCP-compatible IDEs, Farnsworth exposes the following resources:

- `farnsworth://memory/recent`: View active context
- `farnsworth://proactive/suggestions`: Get anticipatory help
- `farnsworth://system/health`: Check system status

## Best Practices

1. **Tagging**: Use consistent tags when storing memories to improve recall accuracy.
2. **Feedback**: Regularly use `delegate_task` with feedback enabled to help the system evolve.
3. **Context**: Keep active context windows clean by archiving old tasks.
