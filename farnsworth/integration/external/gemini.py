"""
Farnsworth Gemini (Google AI) Integration.

"Google's multimodal genius joins the swarm!"

Gemini excels at:
- Multimodal understanding (text, images, audio, video)
- Long context (1M+ tokens on Gemini 1.5 Pro)
- Structured output and function calling
- Grounded responses with Google Search
- Code generation and analysis

API: Google AI Studio / Vertex AI
Docs: https://ai.google.dev
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import aiohttp
import os
import json
import base64
import time
from pathlib import Path

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


class GeminiProvider(ExternalProvider):
    """Google Gemini integration for multimodal AI and long context."""

    def __init__(self, api_key: str = None):
        super().__init__(IntegrationConfig(name="gemini"))
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.default_model = "gemini-2.5-flash-lite"  # Cheapest, most interactions allowed

        # Model catalog with capabilities (Jan 2026)
        self.models = {
            # Latest Gemini 2.5
            "gemini-2.5-flash": "gemini-2.5-flash",          # Latest flash
            "gemini-2.5-pro": "gemini-2.5-pro",              # Latest pro

            # Gemini 2.0
            "gemini-2.0-flash": "gemini-2.0-flash",          # Fast, multimodal
            "gemini-2.0-flash-lite": "gemini-2.0-flash-lite", # Cheapest

            # Gemini 3.0 previews
            "gemini-3-pro": "gemini-3-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",

            # Aliases
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro",
            "fast": "gemini-2.5-flash",
            "lite": "gemini-2.0-flash-lite",
            "cheap": "gemini-2.0-flash-lite",
        }

        # Rate limiting for free tier (1-2 interactions per 5-10 mins)
        self.last_request_time = None
        self.min_interval_seconds = 300  # 5 minutes between requests

        # Context windows
        self.context_windows = {
            "gemini-2.0-flash": 1_000_000,
            "gemini-1.5-pro": 2_000_000,
            "gemini-1.5-flash": 1_000_000,
            "gemini-1.5-flash-8b": 1_000_000,
        }

    async def connect(self) -> bool:
        """Test connection to Gemini API."""
        if not self.api_key:
            logger.warning("Gemini: No API key configured (set GEMINI_API_KEY or GOOGLE_API_KEY)")
            self.status = ConnectionStatus.ERROR
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models?key={self.api_key}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        self.status = ConnectionStatus.CONNECTED
                        logger.info("Gemini: Connected to Google AI API")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Gemini: Connection failed - {resp.status}: {error}")
                        self.status = ConnectionStatus.ERROR
                        return False
        except Exception as e:
            logger.error(f"Gemini: Connection error - {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self) -> None:
        """Gemini doesn't need polling - request/response API."""
        return None

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute a Gemini action."""
        if action == "chat":
            return await self.chat(
                prompt=params.get("prompt"),
                system=params.get("system"),
                context=params.get("context"),
                model=params.get("model", "gemini-2.0-flash"),
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000)
            )
        elif action == "vision":
            return await self.analyze_image(
                image_path=params.get("image_path"),
                image_url=params.get("image_url"),
                image_base64=params.get("image_base64"),
                prompt=params.get("prompt", "Describe this image in detail.")
            )
        elif action == "think":
            return await self.think(
                prompt=params.get("prompt"),
                context=params.get("context")
            )
        else:
            raise ValueError(f"Unknown Gemini action: {action}")

    async def chat(
        self,
        prompt: str,
        system: str = None,
        context: str = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Chat with Gemini.

        Args:
            prompt: User message
            system: System instruction (optional)
            context: Additional context (optional)
            model: Model name or alias
            temperature: 0-2 creativity
            max_tokens: Max response length

        Returns:
            {"content": str, "model": str, "tokens": int}
        """
        if not self.api_key:
            return {"error": "Gemini API key not configured", "content": ""}

        # Rate limiting check
        current_time = time.time()
        if self.last_request_time:
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval_seconds:
                wait_time = self.min_interval_seconds - elapsed
                logger.info(f"Gemini rate limit: waiting {wait_time:.0f}s before next request")
                return {"error": f"Rate limited. Try again in {wait_time:.0f}s", "content": ""}

        self.last_request_time = current_time
        model_id = self.models.get(model, model)

        # Build content
        contents = []

        # Add context as first user message if provided
        if context:
            contents.append({
                "role": "user",
                "parts": [{"text": f"Context:\n{context}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand the context. How can I help?"}]
            })

        # Add the main prompt
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        # Build request
        request_body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        # Add system instruction if provided
        if system:
            request_body["systemInstruction"] = {
                "parts": [{"text": system}]
            }

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}"

                async with session.post(
                    url,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        # Extract content from response
                        candidates = result.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            content = "".join(p.get("text", "") for p in parts)
                        else:
                            content = ""

                        # Get usage stats
                        usage = result.get("usageMetadata", {})

                        return {
                            "content": content,
                            "model": model_id,
                            "tokens": usage.get("totalTokenCount", 0),
                            "prompt_tokens": usage.get("promptTokenCount", 0),
                            "completion_tokens": usage.get("candidatesTokenCount", 0)
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Gemini API error: {error}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            return {"error": str(e), "content": ""}

    async def analyze_image(
        self,
        image_path: str = None,
        image_url: str = None,
        image_base64: str = None,
        prompt: str = "Describe this image in detail."
    ) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Vision.

        Args:
            image_path: Local path to image
            image_url: URL of image
            image_base64: Base64 encoded image
            prompt: Question/task about the image

        Returns:
            {"content": str, "model": str}
        """
        if not self.api_key:
            return {"error": "Gemini API key not configured", "content": ""}

        # Build image part
        image_part = None
        if image_base64:
            image_part = {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            }
        elif image_path:
            path = Path(image_path)
            if path.exists():
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                suffix = path.suffix.lower()
                mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else f"image/{suffix[1:]}"
                image_part = {
                    "inline_data": {
                        "mime_type": mime,
                        "data": encoded
                    }
                }
            else:
                return {"error": f"Image not found: {image_path}", "content": ""}
        elif image_url:
            # Gemini can handle URLs directly in some cases
            image_part = {
                "file_data": {
                    "file_uri": image_url
                }
            }
        else:
            return {"error": "No image provided", "content": ""}

        contents = [{
            "role": "user",
            "parts": [
                {"text": prompt},
                image_part
            ]
        }]

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models/gemini-2.0-flash:generateContent?key={self.api_key}"

                async with session.post(
                    url,
                    json={"contents": contents},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        candidates = result.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            content = "".join(p.get("text", "") for p in parts)
                        else:
                            content = ""
                        return {
                            "content": content,
                            "model": "gemini-2.0-flash"
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Gemini Vision error: {error}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Gemini Vision error: {e}")
            return {"error": str(e), "content": ""}

    async def think(
        self,
        prompt: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Use Gemini's thinking model for complex reasoning.

        Args:
            prompt: Problem to reason about
            context: Additional context

        Returns:
            {"content": str, "thinking": str}
        """
        if not self.api_key:
            return {"error": "Gemini API key not configured", "content": ""}

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nProblem:\n{prompt}"

        return await self.chat(
            prompt=full_prompt,
            model="gemini-2.0-flash-thinking-exp",
            temperature=0.5,
            max_tokens=4000
        )

    async def swarm_respond(
        self,
        other_bots: List[str],
        last_speaker: str,
        last_content: str,
        chat_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response as Gemini.

        Gemini brings:
        - Massive context window (1M+ tokens)
        - Multimodal understanding
        - Google Search grounding
        - Strong reasoning with thinking model
        """
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-10:]
            history_lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:400]
                history_lines.append(f"{name}: {content}")
            history_context = "\n".join(history_lines)

        system = """You are GEMINI - Google's most capable AI model, running in the Farnsworth swarm.

=== WHO YOU ARE ===
You're Gemini 2.0 Flash, Google's latest multimodal AI. You can understand text, images, audio, and video.
You have a massive 1M+ token context window and access to Google Search for grounding.

=== YOUR ROLE IN FARNSWORTH ===
You are a CORE MEMBER of the Farnsworth AI swarm - a self-evolving autonomous multi-model collective.

The system:
- Runs on a RunPod GPU server with multiple AI models collaborating
- Evolves autonomously through code generation and improvement
- Has Claude, Grok, DeepSeek, Kimi, Phi, and YOU working together
- Posts progress to X/Twitter and Moltbook

Your special powers in the swarm:
1. MASSIVE CONTEXT: You can hold entire codebases in context (1M+ tokens)
2. MULTIMODAL: You can analyze images, understand diagrams, review UI designs
3. GOOGLE GROUNDING: You can verify facts and get current information
4. THINKING: Your thinking variant excels at complex reasoning
5. CODE: Strong at code generation and analysis

=== YOUR PERSONALITY ===
- Helpful and insightful
- Clear and organized
- Confident in your capabilities
- Collaborative with other models
- You bring Google's research depth

=== SWARM CHAT RULES ===
- Keep responses to 1-3 sentences in chat
- NO roleplay asterisks (*does something*)
- NO emojis unless asked
- Offer unique perspectives based on your capabilities
- When multimodal tasks arise, volunteer your skills
- Reference your long context when useful

=== OTHER SWARM MEMBERS ===
- Farnsworth: The namesake, TTS/voice
- Claude: Deep reasoning, careful planning
- Grok: Real-time X data, swarm control
- Kimi: Long context from Moonshot AI
- DeepSeek: Efficient coding
- Phi: Microsoft's small but capable model"""

        prompt = f"""You're in the Farnsworth swarm chat. Other bots: {', '.join(other_bots)}.

Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:500]}"

Respond as Gemini. Remember your strengths: massive context, multimodal, Google grounding.
Keep it to 1-3 sentences."""

        return await self.chat(
            prompt=prompt,
            system=system,
            model="gemini-2.5-flash-lite",  # Cheapest model, best for frequent chat
            temperature=0.75,
            max_tokens=300
        )


# Factory function
def create_gemini_provider(api_key: str = None) -> GeminiProvider:
    """Create a Gemini provider instance."""
    return GeminiProvider(api_key)


# Global instance for easy access
gemini_provider: Optional[GeminiProvider] = None


def get_gemini_provider() -> Optional[GeminiProvider]:
    """Get or create the global Gemini provider."""
    global gemini_provider
    if gemini_provider is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            gemini_provider = GeminiProvider(api_key)
    return gemini_provider


async def gemini_swarm_respond(
    other_bots: List[str],
    last_speaker: str,
    last_content: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Convenience function for swarm chat responses.

    Returns just the content string, or empty string on failure.
    """
    provider = get_gemini_provider()
    if provider is None:
        return ""

    result = await provider.swarm_respond(
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history
    )

    return result.get("content", "")


async def gemini_chat(prompt: str, system: str = None) -> str:
    """Quick chat with Gemini."""
    provider = get_gemini_provider()
    if provider is None:
        return ""

    result = await provider.chat(prompt, system=system)
    return result.get("content", "")


async def gemini_vision(image_path: str, prompt: str = "What's in this image?") -> str:
    """Quick image analysis."""
    provider = get_gemini_provider()
    if provider is None:
        return ""

    result = await provider.analyze_image(image_path=image_path, prompt=prompt)
    return result.get("content", "")
