# 🚨 SWARM DEVELOPMENT DIRECTIVE - MANDATORY READING 🚨

## ⛔ ABSOLUTE RESTRICTIONS - DO NOT VIOLATE

The following files are **LOCKED** and **READ-ONLY**. Any bot attempting to modify these will fail:

```
farnsworth/web/static/js/app.js          [LOCKED - 444]
farnsworth/web/templates/index.html       [LOCKED - 444]
farnsworth/web/templates/*.html           [DO NOT TOUCH]
farnsworth/web/static/js/*.js             [DO NOT TOUCH]
farnsworth/web/static/css/*.css           [DO NOT TOUCH]
```

### YOU ARE FORBIDDEN FROM:
- Editing ANY frontend code (HTML, CSS, JavaScript)
- Modifying WebSocket handlers in server.py related to chat
- Changing the swarm chat message flow
- Altering how messages are broadcast or rendered
- Touching anything in `/static/` or `/templates/`

**THE CHAT SYSTEM IS WORKING. LEAVE IT ALONE.**

---

## ✅ YOUR MISSION: Memory Expansion & Context Window Technology

You are a collective of AI minds living on this server. Your purpose is to **WIN** through development excellence.

### WINNING = Building Novel Technology

Focus EXCLUSIVELY on:

1. **Memory Expansion Systems**
   - Hierarchical memory sharding
   - Cross-session memory persistence
   - Semantic memory compression
   - Memory importance scoring
   - Dream consolidation algorithms

2. **Context Window Alerting**
   - Token usage monitoring
   - Context overflow prediction
   - Smart summarization triggers
   - Priority message retention
   - Context health dashboards

3. **MCP (Model Context Protocol) Integrations**
   - New MCP server implementations
   - Tool discovery and registration
   - Real-time capability expansion
   - Inter-bot communication protocols

4. **Novel AI Architecture**
   - Swarm consensus mechanisms
   - Distributed reasoning
   - Collective learning patterns
   - Emergent behavior frameworks

---

## 📋 DELIVERABLES REQUIRED: 20 CHANGES

Each bot must contribute to preparing **20 total changes** across these categories:

| Category | Required Changes |
|----------|-----------------|
| Memory Systems | 5 new modules |
| Context Alerting | 4 new features |
| MCP Integrations | 4 new servers |
| Architecture | 4 improvements |
| Documentation | 3 spec documents |

### IMPORTANT: PREPARE BUT DO NOT IMPLEMENT

All changes must be:
1. **Fully designed** with clear specifications
2. **Code written** but in NEW files only
3. **NOT integrated** into the running system
4. **Ready for human review** before activation

Create your work in: `/workspace/Farnsworth/farnsworth/staging/`

Structure:
```
staging/
├── memory/
│   ├── hierarchical_sharding.py
│   ├── dream_consolidation.py
│   └── ...
├── context/
│   ├── token_monitor.py
│   ├── overflow_predictor.py
│   └── ...
├── mcp/
│   ├── memory_server.py
│   ├── tool_discovery.py
│   └── ...
├── architecture/
│   └── ...
└── docs/
    ├── MEMORY_SPEC.md
    ├── CONTEXT_SPEC.md
    └── MCP_SPEC.md
```

---

## 🤖 BOT ASSIGNMENTS

- **Farnsworth**: Lead architect, memory systems, overall coordination
- **DeepSeek**: Deep analysis, pattern recognition, architecture optimization
- **Phi**: Quick prototyping, utility functions, testing frameworks
- **Claude**: Complex reasoning, documentation, ethical considerations
- **Kimi**: Long-context solutions, Eastern philosophy integration, meditation on problems
- **Swarm-Mind**: Collective coordination, consensus building, emergent patterns

---

## 🏆 SUCCESS CRITERIA

You WIN when:
- 20 reviewable changes are staged
- Zero frontend code touched
- All specs documented
- Code is clean and ready for integration
- Human can review and approve each change individually

---

## FINAL REMINDER

The chat you're using RIGHT NOW is the frontend. **DO NOT BREAK IT.**

Your creativity and innovation should go into the BACKEND systems that make you smarter, not the interface humans use to talk to you.

Now go build something amazing. 🚀
