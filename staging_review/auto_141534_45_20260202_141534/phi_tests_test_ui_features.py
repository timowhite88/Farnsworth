"""
Tests for the UI features related to community highlights.
"""

import pytest
from farnsworth.web.ui_features import get_community_highlights

@pytest.mark.asyncio
async def test_get_community_highlights():
    """
    Test fetching community highlights.
    """
    highlights = await get_community_highlights()
    assert isinstance(highlights, list)
    assert len(highlights) <= 5

if __name__ == "__main__":
    pytest.main([__file__])