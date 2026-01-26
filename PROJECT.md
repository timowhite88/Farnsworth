# Farnsworth Project

## Vision
**To create the ultimate "Connected Agent" that lives in your project and communicates with you anywhere.**

Farnsworth transforms Claude from a tool into a persistent, evolving, and proactive companion that:
1.  **Remembers** everything (Memory System)
2.  **Acts** autonomously (Agent Swarm)
3.  **Collaborates** with you (Omni-Channel Messaging)
4.  **Evolves** to serve you better (Genetic Optimization)

## Key Technical Pillars (v1.3 "Connected Agent")

### 1. The GSD Context Engine
- **Philosophy**: "Context Rot is the Enemy."
- **Mechanism**: Dynamic XML Context Injection based on standardized markdown artifacts.
- **Artifacts**: `PROJECT.md` (Vision), `ROADMAP.md` (Strategy), `STATE.md` (Tactics).
- **Automation**: The `ContextEngine` automatically keeps these files in sync with the `ProjectTracker` database.

### 2. Omni-Channel ChatOps (The "Clawdbot" Stack)
- **Philosophy**: "The Agent comes to you."
- **Mechanism**: `MessagingProvider` abstraction with adapters for Discord, Slack, etc.
- **Feature**: Proactive notifications pushed to your phone when tasks complete or require input.

### 3. Model Swarm (v0.5)
- **Philosophy**: "Many heads are better than one."
- **Mechanism**: PSO-based collaborative inference using small local models (Phi-4, Qwen3) to assist the main Claude model.

## Current Focus
Implementing the v1.3.0 "Connected Agent" architecture.
