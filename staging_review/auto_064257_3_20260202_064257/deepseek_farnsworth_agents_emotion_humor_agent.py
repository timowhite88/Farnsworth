"""
Agent for evaluating the effect of humor on emotional understanding.
"""

import asyncio
from typing import Dict, Any

class EmotionHumorAgent:
    def __init__(self):
        self.humor_analyzer = HumorAnalysis()

    async def evaluate_humor_effect(self, text: str) -> Dict[str, Any]:
        """
        Evaluates the effect of humor in a given text on emotional understanding.

        :param text: The input text containing potential humor.
        :return: A dictionary with analysis and emotional impact results.
        """
        try:
            humor_analysis = await self.humor_analyzer.analyze_humor_content(text)
            emotional_impact = await self.humor_analyzer.get_emotional_impact(humor_analysis)

            return {
                "humor_analysis": humor_analysis,
                "emotional_impact": emotional_impact
            }
        except Exception as e:
            logger.error(f"Error evaluating humor effect: {e}")
            raise

if __name__ == "__main__":
    # Test code
    async def test():
        agent = EmotionHumorAgent()
        text = "Why don't scientists trust atoms? Because they make up everything!"
        result = await agent.evaluate_humor_effect(text)
        print(f"Result: {result}")

    asyncio.run(test())