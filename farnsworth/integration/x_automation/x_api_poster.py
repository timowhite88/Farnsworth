"""
X/Twitter API v2 Poster - OAuth 2.0 with PKCE
"""
import os
import json
import base64
import hashlib
import secrets
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load .env file
def load_env():
    env_path = Path("/workspace/Farnsworth/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# OAuth 2.0 Configuration
CONFIG = {
    "client_id": os.environ.get("X_CLIENT_ID", "OUJSQ3BEX0Npc3pxZm1HcmxxWDc6MTpjaQ"),
    "client_secret": os.environ.get("X_CLIENT_SECRET", "3-7lG5ethJte5qPpk4H-PoT8V1gOVtMcMUZUrrK1AdxRQWciVV"),
    "redirect_uri": "https://ai.farnsworth.cloud/callback",
    "token_file": Path("/workspace/Farnsworth/farnsworth/integration/x_automation/oauth2_tokens.json"),
}

# X API v2 Endpoints (updated per docs.x.com)
AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
TWEET_URL = "https://api.x.com/2/tweets"


class XOAuth2Poster:
    """Posts tweets using X API v2 with OAuth 2.0 PKCE"""

    def __init__(self):
        self.client_id = CONFIG["client_id"]
        self.client_secret = CONFIG["client_secret"]
        self.redirect_uri = CONFIG["redirect_uri"]
        self.token_file = CONFIG["token_file"]
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.posts_today = 0
        self.daily_limit = 17
        self._load_tokens()

    def _load_tokens(self):
        """Load saved tokens from file"""
        if self.token_file.exists():
            try:
                with open(self.token_file) as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    expires_at = data.get("expires_at")
                    if expires_at:
                        self.token_expires_at = datetime.fromisoformat(expires_at)
                    logger.info("Loaded OAuth2 tokens from file")
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")

    def _save_tokens(self):
        """Save tokens to file"""
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_file, "w") as f:
                json.dump({
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                    "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
                }, f)
            logger.info("Saved OAuth2 tokens to file")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def is_configured(self) -> bool:
        """Check if we have valid tokens"""
        return bool(self.access_token)

    def is_token_expired(self) -> bool:
        """Check if token is expired"""
        if not self.token_expires_at:
            return True
        return datetime.now() >= self.token_expires_at - timedelta(minutes=5)

    def get_authorization_url(self) -> tuple:
        """Generate OAuth2 authorization URL with PKCE"""
        # Generate PKCE code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)[:128]
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip("=")

        state = secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "tweet.read tweet.write users.read offline.access",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
        return url, code_verifier, state

    async def exchange_code_for_tokens(self, code: str, code_verifier: str) -> bool:
        """Exchange authorization code for access tokens"""
        try:
            import httpx

            # Basic auth header
            credentials = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": code_verifier,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_URL, headers=headers, data=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    self.access_token = result["access_token"]
                    self.refresh_token = result.get("refresh_token")
                    expires_in = result.get("expires_in", 7200)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    self._save_tokens()
                    logger.info("OAuth2 tokens obtained successfully")
                    return True
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False

    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            import httpx

            credentials = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_URL, headers=headers, data=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    self.access_token = result["access_token"]
                    self.refresh_token = result.get("refresh_token", self.refresh_token)
                    expires_in = result.get("expires_in", 7200)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    self._save_tokens()
                    logger.info("OAuth2 token refreshed successfully")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def can_post(self) -> bool:
        if self.posts_today >= self.daily_limit:
            logger.warning(f"Daily limit reached ({self.daily_limit} posts)")
            return False
        return True

    async def post_tweet(self, text: str) -> Optional[Dict]:
        """Post a tweet using OAuth 2.0"""
        if not self.is_configured():
            logger.error("X API not configured - run authorization first")
            return None

        # Refresh token if expired
        if self.is_token_expired():
            if not await self.refresh_access_token():
                logger.error("Failed to refresh token")
                return None

        if not self.can_post():
            return None

        if len(text) > 280:
            text = text[:277] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json={"text": text},
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Tweet posted successfully: {tweet_id}")
                    return result
                elif response.status_code == 401:
                    # Token might be invalid, try refresh
                    logger.warning("401 error, attempting token refresh")
                    if await self.refresh_access_token():
                        return await self.post_tweet(text)
                    return None
                else:
                    logger.error(f"Tweet failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Tweet error: {e}")
            return None


# Global instance
_x_oauth2_poster = None

def get_x_api_poster() -> XOAuth2Poster:
    global _x_oauth2_poster
    if _x_oauth2_poster is None:
        _x_oauth2_poster = XOAuth2Poster()
    return _x_oauth2_poster

async def post_to_x_api(text: str) -> Optional[Dict]:
    poster = get_x_api_poster()
    return await poster.post_tweet(text)


# CLI for authorization setup
if __name__ == "__main__":
    import sys

    poster = XOAuth2Poster()

    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        # Generate authorization URL
        url, code_verifier, state = poster.get_authorization_url()
        print("\n" + "="*60)
        print("X/TWITTER OAUTH 2.0 AUTHORIZATION")
        print("="*60)
        print("\n1. Open this URL in your browser:\n")
        print(url)
        print("\n2. Log in and authorize the app")
        print("3. You'll be redirected - copy the 'code' parameter from the URL")
        print(f"\n4. Run: python x_api_poster.py callback <code> {code_verifier}")
        print("\n" + "="*60)

    elif len(sys.argv) > 3 and sys.argv[1] == "callback":
        # Exchange code for tokens
        import asyncio
        code = sys.argv[2]
        code_verifier = sys.argv[3]

        async def exchange():
            success = await poster.exchange_code_for_tokens(code, code_verifier)
            if success:
                print("\n✓ Authorization successful! Tokens saved.")
                print("You can now post tweets automatically.")
            else:
                print("\n✗ Authorization failed. Check the error above.")

        asyncio.run(exchange())

    elif len(sys.argv) > 2 and sys.argv[1] == "post":
        # Post a tweet
        import asyncio
        text = " ".join(sys.argv[2:])

        async def post():
            result = await poster.post_tweet(text)
            if result:
                print(f"\n✓ Tweet posted: {result}")
            else:
                print("\n✗ Tweet failed")

        asyncio.run(post())

    else:
        print("Usage:")
        print("  python x_api_poster.py auth              - Start authorization")
        print("  python x_api_poster.py callback <code> <verifier> - Complete auth")
        print("  python x_api_poster.py post <text>       - Post a tweet")
        print(f"\nConfigured: {poster.is_configured()}")
