"""
Module for fetching and displaying community highlights from recent deliberations.
"""

import asyncio
from typing import List, Dict
from loguru import logger

async def get_community_highlights() -> List[Dict]:
    """
    Fetch recent deliberation highlights from the collective system.

    Returns:
        List of dictionaries containing highlight information.
    """
    try:
        from farnsworth.core.collective import fetch_recent_deliberations
        
        highlights = await fetch_recent_deliberations(limit=5)
        logger.info("Successfully fetched community highlights.")
        return highlights
    except Exception as e:
        logger.error(f"Error fetching community highlights: {e}")
        return []

# filename: server.py
"""
FastAPI server setup for serving community highlights.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from farnsworth.web.ui_features import get_community_highlights

router = APIRouter()
templates = Jinja2Templates(directory="farnsworth/web/templates")

@router.get("/community-highlights")
async def community_highlights_endpoint(request: Request):
    """
    Endpoint to fetch and render community highlights.

    Returns:
        Rendered HTML page with community highlights.
    """
    try:
        highlights = await get_community_highlights()
        return templates.TemplateResponse(
            "community_highlights.html",
            {"request": request, "highlights": highlights}
        )
    except Exception as e:
        logger.error(f"Error in community_highlights_endpoint: {e}")
        return templates.TemplateResponse("error.html", {"request": request})

# filename: tests/test_ui_features.py
"""
Unit tests for the UI features module.
"""

import pytest
from farnsworth.web.ui_features import get_community_highlights

@pytest.mark.asyncio
async def test_get_community_highlights():
    """
    Test fetching community highlights returns a list of up to 5 items.
    """
    highlights = await get_community_highlights()
    assert isinstance(highlights, list)
    assert len(highlights) <= 5

# filename: farnsworth/web/templates/community_highlights.html
"""
HTML template for displaying community highlights.
"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Community Highlights</title>
</head>
<body>
    <h1>Recent Community Highlights</h1>
    {% if highlights %}
        <ul>
            {% for highlight in highlights %}
                <li>{{ highlight['title'] }}: {{ highlight['summary'] }}</li>
            {% endfor %}
        </ul>
    {% else %}
        <p>No recent highlights available.</p>
    {% endif %}
</body>
</html>

# filename: farnsworth/web/templates/error.html
"""
HTML template for displaying error messages.
"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Error</title>
</head>
<body>
    <h1>An Error Occurred</h1>
    <p>Sorry, something went wrong. Please try again later.</p>
</body>
</html>