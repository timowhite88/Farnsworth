"""
Unit tests for humor analysis and its influence on emotional understanding.
"""

import pytest
from farnsworth.core.humor_analysis import analyze_humor_content, get_emotional_impact

@pytest.mark.asyncio
async def test_analyze_humor_content():
    text = "Why don't scientists trust atoms? Because they make up everything!"
    result = await analyze_humor_content(text)
    
    assert isinstance(result, dict), "Result should be a dictionary."
    assert all(isinstance(value, float) for value in result.values()), "All scores should be floats."

@pytest.mark.asyncio
async def test_get_emotional_impact():
    analysis = {
        "joy": 0.7,
        "surprise": 0.5,
        "sadness": -0.2,
        "anger": -0.1
    }
    
    impact = await get_emotional_impact(analysis)
    
    assert impact in ["positive", "negative", "neutral"], "Impact should be one of the valid states."

if __name__ == "__main__":
    pytest.main()