

# ============================================
# X/TWITTER OAUTH2 CALLBACK
# ============================================

_oauth_state = {}

@app.get("/x/auth")
async def x_auth_start():
    """Start X OAuth2 authorization"""
    from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
    poster = get_x_api_poster()
    url, code_verifier, state = poster.get_authorization_url()
    _oauth_state[state] = code_verifier
    return {"auth_url": url, "state": state}

@app.get("/callback")
async def oauth_callback(request: Request):
    """Handle OAuth2 callback from X"""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    if error:
        return HTMLResponse(f"<h1>Authorization Failed</h1><p>{error}</p>")
    if not code:
        return HTMLResponse("<h1>Missing Code</h1>")
    code_verifier = _oauth_state.get(state)
    if not code_verifier:
        return HTMLResponse("<h1>Invalid State</h1><p>Try again from /x/auth</p>")
    from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
    poster = get_x_api_poster()
    success = await poster.exchange_code_for_tokens(code, code_verifier)
    if success:
        del _oauth_state[state]
        return HTMLResponse("<h1>Success!</h1><p>X posting enabled.</p><a href='/'>Return</a>")
    return HTMLResponse("<h1>Token Exchange Failed</h1>")

# ============================================
# SOCIAL MEDIA MANAGER AUTO-START
# ============================================
from farnsworth.integration.x_automation.social_manager import start_social_manager, get_social_manager

@app.on_event("startup")
async def start_social_posting():
    """Start continuous social media posting"""
    try:
        await start_social_manager()
        logger.info("Social Media Manager started")
    except Exception as e:
        logger.error(f"Failed to start social manager: {e}")

@app.get("/api/social/status")
async def social_status():
    """Get social posting status"""
    from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
    poster = get_x_api_poster()
    manager = get_social_manager()
    return {
        "x_configured": poster.is_configured(),
        "x_needs_auth": not poster.is_configured(),
        "auth_url": "https://ai.farnsworth.cloud/x/auth",
        "manager": manager.get_status()
    }

@app.post("/api/social/post-now")
async def force_post():
    """Force an immediate post"""
    manager = get_social_manager()
    await manager.post_cycle()
    return {"status": "posted", "info": manager.get_status()}
