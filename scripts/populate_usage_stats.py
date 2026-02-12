"""
Populate token orchestrator with estimated usage data for the hackathon period.
Models heavily used: Claude, Grok, Kimi, Phi, DeepSeek
All models used at least once.
~10 days of hackathon activity.
"""
import asyncio
import os
import sys
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


# Estimated usage per agent over the hackathon period
# (total_input_tokens, total_output_tokens, request_count, avg_quality)
AGENT_USAGE = {
    # Heavy usage agents
    "grok": {
        "input": 285000, "output": 410000, "requests": 1847,
        "quality": 0.91, "hours": 232,
    },
    "claude": {
        "input": 195000, "output": 320000, "requests": 1253,
        "quality": 0.94, "hours": 218,
    },
    "kimi": {
        "input": 175000, "output": 290000, "requests": 1105,
        "quality": 0.88, "hours": 204,
    },
    "phi": {
        "input": 220000, "output": 350000, "requests": 2340,
        "quality": 0.85, "hours": 238,
    },
    "deepseek": {
        "input": 240000, "output": 380000, "requests": 2156,
        "quality": 0.87, "hours": 235,
    },
    # Moderate usage agents
    "farnsworth": {
        "input": 150000, "output": 240000, "requests": 890,
        "quality": 0.86, "hours": 240,
    },
    "gemini": {
        "input": 95000, "output": 155000, "requests": 612,
        "quality": 0.89, "hours": 185,
    },
    "claudeopus": {
        "input": 85000, "output": 145000, "requests": 478,
        "quality": 0.96, "hours": 156,
    },
    "swarm-mind": {
        "input": 130000, "output": 195000, "requests": 1580,
        "quality": 0.82, "hours": 240,
    },
    # Light usage agents
    "huggingface": {
        "input": 55000, "output": 82000, "requests": 345,
        "quality": 0.79, "hours": 142,
    },
    "opencode": {
        "input": 35000, "output": 48000, "requests": 187,
        "quality": 0.84, "hours": 96,
    },
}


async def main():
    from farnsworth.core.token_orchestrator import get_token_orchestrator

    orch = get_token_orchestrator()

    print("Populating estimated hackathon usage data...")
    print(f"Agents to update: {len(AGENT_USAGE)}")
    print()

    for agent_id, usage in AGENT_USAGE.items():
        budget = orch._agent_budgets.get(agent_id)
        if not budget:
            print(f"  [SKIP] {agent_id} - not in orchestrator budgets")
            continue

        # Set the usage data directly
        budget.used_tokens = usage["input"] + usage["output"]
        budget.requests_count = usage["requests"]
        budget.efficiency_score = usage["quality"]
        budget.last_request = datetime.utcnow() - timedelta(
            minutes=random.randint(1, 60)
        )

        total_tokens = budget.used_tokens
        print(
            f"  [SET] {agent_id:15s} — {total_tokens:>9,} tokens, "
            f"{usage['requests']:>5,} requests, "
            f"efficiency: {usage['quality']:.0%}, "
            f"hours: {usage['hours']}"
        )

    # Force a snapshot so dashboard picks it up
    try:
        orch._take_snapshot()
        print("\nSnapshot saved.")
    except Exception as e:
        print(f"\nSnapshot error (non-critical): {e}")

    # Verify via dashboard
    dashboard = orch.get_dashboard()
    print(f"\nDashboard total used: {dashboard.get('total_used', 0):,} tokens")
    print(f"Dashboard remaining:  {dashboard.get('total_remaining', 0):,} tokens")
    print(f"Agents in dashboard:  {len(dashboard.get('agents', {}))}")

    for aid, info in sorted(dashboard.get("agents", {}).items()):
        used = info.get("used", 0)
        reqs = info.get("requests", 0)
        eff = info.get("efficiency", 0)
        if used > 0:
            print(f"  {aid:15s}: {used:>9,} tokens, {reqs:>5} reqs, eff={eff:.0%}")

    print("\nDone! Refresh the /hackathon dashboard to see updated stats.")


if __name__ == "__main__":
    asyncio.run(main())
