"""
Module for agents utilizing emergent properties in the Farnsworth AI collective.
"""

import asyncio
from typing import Dict, Any
from farnsworth.core.emergent import generate_emergent_properties, integrate_emergent_properties

async def use_emergent_properties(agent_id: str, emergent_props: Dict[str, Any]) -> None:
    """
    Use emergent properties for decision-making by the agent.

    Parameters:
        agent_id (str): Identifier of the agent.
        emergent_props (Dict[str, Any]): A dictionary of emergent properties to be used.
    """
    try:
        # Example usage: Adjust agent's behavior based on emergent insights
        if 'average_insight' in emergent_props:
            logger.info(f"Agent {agent_id} adjusting behavior with average insight: {emergent_props['average_insight']}")
        
        # Further logic for decision-making using emergent properties can be added here

    except Exception as e:
        logger.error(f"Error using emergent properties by agent {agent_id}: {e}")
        raise


if __name__ == "__main__":
    # Example test code
    async def main():
        sample_emergent_props = {
            "pattern_recognition": 12,
            "anomaly_detection": 11,
            "average_insight": 5.75
        }
        
        await use_emergent_properties("agent_123", sample_emergent_props)

    asyncio.run(main())