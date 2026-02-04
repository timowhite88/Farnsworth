# Development Plan

Task: hey everyone! good news, i’ve been tinkering and thinking about what makes our community tick

To implement a simple UI feature that showcases what makes our community tick, we will create an interface for displaying recent discussions or highlights from the collective deliberation system. This involves adding new files and functions to manage and display this data.

### Implementation Plan

#### 1. Files to Create

- **farnsworth/web/ui_features.py**: This file will contain the logic for fetching and processing community highlights.
  
- **farnsworth/web/templates/community_highlights.html**: An HTML template to render the UI component displaying community highlights.

#### 2. Functions to Implement

**In `farnsworth/web/ui_features.py`:**

```python
from typing import List, Dict
from farnsworth.core.collective import fetch_recent_deliberations

async def get_community_highlights() -> List[Dict]:
    """
    Fetch recent deliberation highlights from the collective system.
    
    Returns:
        List of dictionaries containing highlight information.
    """
    return await fetch_recent_deliberations(limit=5)
```

**In `farnsworth/web/server.py`:**

Add a new endpoint to serve community highlights:

```python
from fastapi import APIRouter, Request
from farnsworth.web.ui_features import get_community_highlights

router = APIRouter()

@router.get("/community-highlights")
async def community_highlights_endpoint(request: Request):
    """
    Endpoint to fetch and render community highlights.
    
    Returns:
        Rendered HTML page with community highlights.
    """
    highlights = await get_community_highlights()
    return templates.TemplateResponse(
        "community_highlights.html",
        {"request": request, "highlights": highlights}
    )
```

#### 3. Imports Required

- **In `farnsworth/web/ui_features.py`:**

```python
from typing import List, Dict
from farnsworth.core.collective import fetch_recent_deliberations
```

- **In `farnsworth/web/server.py`:**

```python
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from farnsworth.web.ui_features import get_community_highlights

templates = Jinja2Templates(directory="farnsworth/web/templates")
```

#### 4. Integration Points

- **Modify `farnsworth/web/server.py`:** Add the new endpoint `/community-highlights` and integrate it with FastAPI.

- **Create a new template file:** `farnsworth/web/templates/community_highlights.html` to render the highlights.

#### 5. Test Commands

1. **Run the FastAPI server:**

   ```bash
   uvicorn farnsworth.web.server:app --reload
   ```

2. **Access the endpoint in a web browser or using curl:**

   - Open `http://localhost:8000/community-highlights` in your browser to view the community highlights.

3. **Verify Functionality:**

   - Ensure that recent deliberations are displayed correctly on the `/community-highlights` page.
   - Check console logs for any errors during server startup or endpoint requests.

4. **Unit Tests (Optional):**

   Create a simple test in `tests/test_ui_features.py`:

   ```python
   import pytest
   from farnsworth.web.ui_features import get_community_highlights

   @pytest.mark.asyncio
   async def test_get_community_highlights():
       highlights = await get_community_highlights()
       assert isinstance(highlights, list)
       assert len(highlights) <= 5
   ```

   Run tests with:

   ```bash
   pytest tests/test_ui_features.py
   ```

This plan outlines the creation of a new UI feature to display community highlights, integrating it into the existing FastAPI server setup.