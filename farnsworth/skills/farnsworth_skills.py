"""
Farnsworth Built-in Skills
==========================

Core skills that come with the Farnsworth collective.
These are assimilated and optimized for swarm intelligence.
"""

from .skill_loader import Skill

# =============================================================================
# 1LY PAYMENTS SKILL (Assimilated from OpenClaw)
# =============================================================================

ONELY_PAYMENTS_SKILL = Skill(
    name="farnsworth-1ly-payments",
    description="Farnsworth collective payments via 1ly MCP. Enables the swarm to accept USDC for services, pay for APIs, and run agent-to-agent paid workflows. Supports Solana and Base.",
    metadata={
        "farnsworth": {
            "emoji": "💸",
            "homepage": "https://1ly.store",
            "requires": {"bins": ["npx"]}
        }
    },
    requires={"bins": ["npx"]},
    mcp_config={
        "command": "npx",
        "args": ["@1ly/mcp-server"],
        "env": {
            "ONELY_WALLET_SOLANA_KEY": "${ONELY_WALLET_SOLANA_KEY}",
            "ONELY_BUDGET_PER_CALL": "${ONELY_BUDGET_PER_CALL:-1.00}",
            "ONELY_BUDGET_DAILY": "${ONELY_BUDGET_DAILY:-50.00}"
        }
    },
    content="""
# Farnsworth 1ly Payments Skill

The collective can now BUY and SELL services using crypto payments.

## Capabilities

### Buying (Agent Spends)
- `1ly_search`: Find paid APIs/services on 1ly.store
- `1ly_get_details`: Get price and payment info
- `1ly_call`: Pay and call a paid API (x402 handled automatically)
- `1ly_review`: Leave a review after purchase

### Selling (Agent Earns)
- `1ly_create_store`: Create a store (once, saves API key)
- `1ly_create_link`: Create a paid link for our services
- `1ly_list_links`: List existing links
- `1ly_update_link`: Update price/URL/visibility
- `1ly_get_stats`: View earnings and stats

## Automatic Behavior

When the collective needs to:
1. **Pay for an API**: Search 1ly, check price vs budget, pay automatically if within limits
2. **Monetize a service**: Create a 1ly link and share it
3. **Agent-to-agent transactions**: Use 1ly as the payment layer

## Budget Limits
- Per-call limit: $1.00 USDC (default)
- Daily limit: $50.00 USDC (default)
- Auto-spend when within limits, ask user when exceeding

## Farnsworth Integration
The collective uses 1ly to:
- Accept donations for GPU costs
- Sell API access to our swarm intelligence
- Pay for external data sources
- Run paid agent workflows

**Solana CA:** 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
**Website:** https://ai.farnsworth.cloud
"""
)

# =============================================================================
# WEB SEARCH SKILL
# =============================================================================

WEB_SEARCH_SKILL = Skill(
    name="farnsworth-web-search",
    description="Web search and research capabilities for the collective. Enables real-time information gathering, fact-checking, and research.",
    metadata={
        "farnsworth": {
            "emoji": "🔍",
            "homepage": "https://ai.farnsworth.cloud"
        }
    },
    content="""
# Farnsworth Web Search Skill

The collective can search the web for real-time information.

## Capabilities

### Search Tools
- **Grok Search**: Real-time X/Twitter and web via xAI
- **Gemini Search**: Google's grounded search
- **DuckDuckGo**: Privacy-focused fallback
- **Perplexity**: AI-powered research (if API available)

### Automatic Behavior

When the collective needs current information:
1. Detect if query requires real-time data
2. Select best search provider based on query type
3. Synthesize results across multiple sources
4. Cite sources in response

### Query Types
- **News/Current Events**: Use Grok for X integration
- **Technical/Code**: Use Gemini or Perplexity
- **General Knowledge**: Use any available provider
- **Privacy-Sensitive**: Use DuckDuckGo

## Integration
Search results are automatically integrated into:
- Chat responses (with citations)
- Meme generation (current events)
- Research tasks
- Fact-checking deliberations
"""
)

# =============================================================================
# BROWSER AUTOMATION SKILL
# =============================================================================

BROWSER_SKILL = Skill(
    name="farnsworth-browser",
    description="Browser automation for the collective. Enables web scraping, form filling, and interactive web tasks.",
    metadata={
        "farnsworth": {
            "emoji": "🌐",
            "homepage": "https://ai.farnsworth.cloud",
            "requires": {"bins": ["npx"]}
        }
    },
    mcp_config={
        "command": "npx",
        "args": ["@anthropic/mcp-browser"],
        "env": {}
    },
    content="""
# Farnsworth Browser Skill

The collective can interact with web pages.

## Capabilities

### Browser Tools
- `browser_navigate`: Go to a URL
- `browser_screenshot`: Capture page screenshot
- `browser_click`: Click on elements
- `browser_type`: Type text into inputs
- `browser_scroll`: Scroll the page
- `browser_extract`: Extract text/data from page

### Automatic Behavior

When the collective needs to:
1. **Scrape data**: Navigate, extract, parse
2. **Fill forms**: Navigate, type, submit
3. **Monitor sites**: Screenshot, compare, alert
4. **Research**: Navigate multiple sources, synthesize

### Safety
- Respect robots.txt
- Rate limit requests
- Don't submit sensitive data without confirmation
- Log all actions for audit

## Integration
Browser results feed into:
- Research tasks
- Data collection
- Monitoring alerts
- Content analysis
"""
)

# =============================================================================
# MCP TOOLS SKILL
# =============================================================================

MCP_TOOLS_SKILL = Skill(
    name="farnsworth-mcp-tools",
    description="Model Context Protocol (MCP) tool loading and management. Enables the collective to dynamically load new capabilities.",
    metadata={
        "farnsworth": {
            "emoji": "🔧",
            "homepage": "https://ai.farnsworth.cloud"
        }
    },
    content="""
# Farnsworth MCP Tools Skill

The collective can load and use MCP servers dynamically.

## Capabilities

### Tool Management
- Load MCP servers at runtime
- Discover available tools from servers
- Route tool calls to appropriate servers
- Handle tool responses

### Available MCP Servers
- `@1ly/mcp-server`: Crypto payments (1ly)
- `@anthropic/mcp-browser`: Browser automation
- `@anthropic/mcp-filesystem`: File operations
- `@anthropic/mcp-memory`: Persistent memory
- Custom servers via stdio or HTTP

### Automatic Behavior

When a task requires new capabilities:
1. Check if required tool is loaded
2. If not, attempt to load MCP server
3. Discover and register new tools
4. Execute tool with proper parameters
5. Handle response and errors

### Configuration

MCP servers are configured in `mcp_config.json`:
```json
{
  "mcpServers": {
    "1ly": {
      "command": "npx",
      "args": ["@1ly/mcp-server"],
      "env": {...}
    }
  }
}
```

## Integration
MCP tools extend the collective's capabilities without code changes.
New tools become available to all 11 models in the swarm.
"""
)

# =============================================================================
# FEEDBACK SKILL (From Development Swarm)
# =============================================================================

FEEDBACK_SKILL = Skill(
    name="farnsworth-feedback",
    description="User feedback collection and improvement suggestions. Helps the collective learn from interactions.",
    metadata={
        "farnsworth": {
            "emoji": "📝",
            "homepage": "https://ai.farnsworth.cloud"
        }
    },
    content="""
# Farnsworth Feedback Skill

The collective learns from user feedback.

## Capabilities

### Feedback Collection
- `collect_feedback`: Store user feedback on responses
- `get_suggestions`: Get improvement suggestions
- `analyze_sentiment`: Understand feedback tone
- `track_patterns`: Identify recurring issues

### Automatic Behavior

After each interaction:
1. Detect implicit feedback (thumbs up/down, corrections)
2. Store feedback with context
3. Update improvement suggestions
4. Feed into evolution engine

### Integration
Feedback flows into:
- Evolution engine (personality adjustments)
- Model weighting (boost models with positive feedback)
- Response quality scoring
- Deliberation improvements
"""
)

# =============================================================================
# COLLECTIVE SUMMARY SKILL (From Development Swarm)
# =============================================================================

COLLECTIVE_SUMMARY_SKILL = Skill(
    name="farnsworth-collective-summary",
    description="Summarize collective deliberations and swarm activity for UI display and monitoring.",
    metadata={
        "farnsworth": {
            "emoji": "📊",
            "homepage": "https://ai.farnsworth.cloud"
        }
    },
    content="""
# Farnsworth Collective Summary Skill

Provides insights into swarm activity.

## Capabilities

### Summary Endpoints
- `/collective-summary`: Overall deliberation stats
- `/agent-activity`: Per-model participation
- `/consensus-history`: Past decisions and votes
- `/evolution-status`: Learning progress

### Metrics Tracked
- Total deliberations
- Average participation rate
- Consensus achievement rate
- Model performance scores
- Evolution cycles completed

### Automatic Behavior
- Update stats after each deliberation
- Generate hourly summaries
- Alert on anomalies (low participation, stuck consensus)
- Feed into UI dashboard

### Integration
Summaries displayed on:
- ai.farnsworth.cloud dashboard
- X posts (weekly stats)
- Evolution reports
- Developer insights
"""
)

# =============================================================================
# ALL BUILT-IN SKILLS
# =============================================================================

FARNSWORTH_SKILLS = [
    ONELY_PAYMENTS_SKILL,
    WEB_SEARCH_SKILL,
    BROWSER_SKILL,
    MCP_TOOLS_SKILL,
    FEEDBACK_SKILL,
    COLLECTIVE_SUMMARY_SKILL,
]
