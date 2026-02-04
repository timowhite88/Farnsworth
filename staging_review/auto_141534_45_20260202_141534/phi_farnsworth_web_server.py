"""
FastAPI server setup including endpoints for community features.
"""

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from farnsworth.web.ui_features import get_community_highlights

app = FastAPI()
templates = Jinja2Templates(directory="farnsworth/web/templates")

@app.get("/community-highlights")
async def community_highlights_endpoint(request: Request):
    """
    Endpoint to fetch and render community highlights.

    Returns:
        Rendered HTML page with community highlights.
    """
    try:
        highlights = await get_community_highlights()
        return templates.TemplateResponse(
            "community_highlights.html", {"request": request, "highlights": highlights}
        )
    except Exception as e:
        logger.error(f"Error rendering community highlights: {e}")
        return templates.TemplateResponse("error.html", {"request": request, "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)