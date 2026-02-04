"""
Module defining an agent that utilizes emergent properties for enhanced decision-making.
"""

import asyncio
from typing import Dict, Any
from loguru import logger
from farnsworth.core.emergent import generate_emergent_properties, integrate_emergent_properties

async def use_emergent_properties(agent_id: str, emergent_props: Dict[str, Any]) -> None:
    """
    Use emergent properties for decision-making by the agent.

    Parameters:
        agent_id (str): Identifier of the agent.
        emergent_props (Dict[str, Any]): A dictionary of emergent properties to be used.
    """
    try:
        logger.info(f"Agent {agent_id} using emergent properties...")
        
        # Example: Log how an agent might use these properties
        decision_factor = emergent_props.get("consensus", 0) * 2 + emergent_props.get("diversity", 1)
        logger.info(f"Decision factor for agent {agent_id}: {decision_factor}")
    except Exception as e:
        logger.error(f"Error using emergent properties for agent {agent_id}: {e}")
        raise

if __name__ == "__main__":
    # Test code
    sample_emergent_props = {
        "consensus": 15.0,
        "diversity": 3
    }
    
    async def test_use_properties():
        await use_emergent_properties("agent_123", sample_emergent_props)

    asyncio.run(test_use_properties())