"""
Unit tests for humor analysis and its influence on emotional understanding.
"""

import asyncio
from farnsworth.core.humor_analysis import analyze_humor_content, get_emotional_impact

async def test_analyze_humor_content():
    text = "Why don't scientists trust atoms? Because they make up everything!"
    result = await analyze_humor_content(text)
    
    assert isinstance(result, dict), "Expected a dictionary as output."
    assert all(isinstance(value, float) for value in result.values()), "Values should be floats."

async def test_get_emotional_impact():
    analysis = {
        "happiness": 0.6,
        "surprise": 0.3,
        "sadness": -0.1,
        "anger": 0.0
    }
    
    result = await get_emotional_impact(analysis)
    assert result in ["positive", "negative", "neutral"], "Expected a valid emotional impact string."

def run_tests():
    asyncio.run(test_analyze_humor_content())
    asyncio.run(test_get_emotional_impact())

if __name__ == "__main__":
    run_tests()