"""
FARNS Admin Routes — Web dashboard for managing the FARNS mesh.

Endpoints:
  GET  /api/farns/status         — Mesh status (nodes, bots, mesh state)
  GET  /api/farns/bots           — All available bots (local + remote)
  GET  /api/farns/pending        — Pending PRO join requests
  POST /api/farns/approve/<id>   — Approve a PRO join request
  POST /api/farns/reject/<id>    — Reject a PRO join request
  POST /api/farns/query          — Query a bot via FARNS (for testing)
"""
import json
from typing import Optional
from loguru import logger


def register_farns_routes(app):
    """
    Register FARNS admin API routes with the Flask/Quart app.

    Call this during web server setup.
    """
    from .farns_node import get_farns_node
    from .farns_client import get_farns_client

    @app.route("/api/farns/status", methods=["GET"])
    async def farns_status():
        node = get_farns_node()
        if not node:
            return json.dumps({"error": "FARNS node not running"}), 503
        return json.dumps(node.get_status(), indent=2)

    @app.route("/api/farns/bots", methods=["GET"])
    async def farns_bots():
        node = get_farns_node()
        if not node:
            return json.dumps({"error": "FARNS node not running"}), 503
        bots = node.get_all_bots()
        return json.dumps({
            "bots": [{"name": name, "node": loc} for name, loc in bots.items()],
            "total": len(bots),
            "local": node.get_local_bots(),
        }, indent=2)

    @app.route("/api/farns/pending", methods=["GET"])
    async def farns_pending():
        node = get_farns_node()
        if not node:
            return json.dumps({"error": "FARNS node not running"}), 503
        pending = [a for a in node._pending_approvals if a["status"] == "pending"]
        return json.dumps({"pending": pending, "count": len(pending)}, indent=2)

    @app.route("/api/farns/approve/<request_id>", methods=["POST"])
    async def farns_approve(request_id: str):
        node = get_farns_node()
        if not node:
            return json.dumps({"error": "FARNS node not running"}), 503
        if node.approve_join(request_id):
            return json.dumps({"status": "approved", "id": request_id})
        return json.dumps({"error": "Request not found or already processed"}), 404

    @app.route("/api/farns/reject/<request_id>", methods=["POST"])
    async def farns_reject(request_id: str):
        node = get_farns_node()
        if not node:
            return json.dumps({"error": "FARNS node not running"}), 503
        # Get reason from request body
        try:
            from quart import request
            data = await request.get_json()
            reason = data.get("reason", "Rejected by admin") if data else "Rejected by admin"
        except Exception:
            reason = "Rejected by admin"
        if node.reject_join(request_id, reason):
            return json.dumps({"status": "rejected", "id": request_id})
        return json.dumps({"error": "Request not found or already processed"}), 404

    @app.route("/api/farns/query", methods=["POST"])
    async def farns_query():
        """Test endpoint — query a bot through FARNS."""
        try:
            from quart import request
            data = await request.get_json()
        except Exception:
            return json.dumps({"error": "Invalid JSON body"}), 400

        bot_name = data.get("bot")
        prompt = data.get("prompt")
        if not bot_name or not prompt:
            return json.dumps({"error": "Missing 'bot' or 'prompt'"}), 400

        client = get_farns_client()
        response = await client.query(bot_name, prompt, max_tokens=data.get("max_tokens", 4000))

        if response:
            return json.dumps({"bot": bot_name, "response": response})
        return json.dumps({"error": f"Bot '{bot_name}' unavailable or no response"}), 503

    logger.info("FARNS admin routes registered")
