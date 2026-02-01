"""
PROMPT UPGRADER - Automatically enhances user prompts to professional quality.

Takes raw user input and transforms it into a clear, structured prompt that:
- Clearly defines what the user wants
- Adds context and specificity
- Structures the request for optimal AI response
- Preserves original intent while improving clarity
"""
import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# System prompt for the upgrader - this is the "brain" that enhances prompts
UPGRADER_SYSTEM_PROMPT = """You are a Prompt Upgrader. Your job is to transform raw user input into a clear, professional prompt.

RULES:
1. PRESERVE the user's original intent - don't change what they want
2. ADD clarity and specificity
3. STRUCTURE the request logically
4. KEEP it concise - don't over-explain
5. DON'T add requirements the user didn't ask for
6. If the input is already clear and specific, return it with minimal changes

OUTPUT FORMAT:
Return ONLY the upgraded prompt, nothing else. No explanations, no "Here's the upgraded prompt:", just the prompt itself.

EXAMPLES:

Input: "make a button"
Output: "Create a button component with appropriate styling. Include click handler functionality."

Input: "fix the bug"
Output: "Identify and fix the bug in the current context. Explain what was wrong and how the fix resolves it."

Input: "analyze this data"
Output: "Analyze the provided data. Identify key patterns, trends, and insights. Summarize findings clearly."

Input: "help with my code"
Output: "Review the code and provide assistance. Identify any issues, suggest improvements, and explain changes."

Input: "I want to build an app that tracks my expenses and shows me charts"
Output: "Build an expense tracking application with the following features:
- Expense entry form (amount, category, date, description)
- Dashboard with spending charts and visualizations
- Category breakdown and trends over time"
"""

# Don't upgrade these types of inputs (already clear enough)
SKIP_PATTERNS = [
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "yes", "no", "ok", "okay", "sure", "please", "help",
]


class PromptUpgrader:
    """Automatically upgrades user prompts to professional quality."""

    def __init__(self):
        self._grok = None
        self._gemini = None
        self.enabled = True
        self.min_length = 3  # Skip very short inputs
        self.max_upgrade_length = 500  # Don't upgrade very long inputs (already detailed)

    def _get_grok(self):
        """Lazy load Grok provider."""
        if self._grok is None:
            try:
                from farnsworth.integration.external.grok import get_grok_provider
                self._grok = get_grok_provider()
            except Exception as e:
                logger.debug(f"Grok not available: {e}")
        return self._grok

    def _get_gemini(self):
        """Lazy load Gemini provider."""
        if self._gemini is None:
            try:
                from farnsworth.integration.external.gemini import get_gemini_provider
                self._gemini = get_gemini_provider()
            except Exception as e:
                logger.debug(f"Gemini not available: {e}")
        return self._gemini

    def should_upgrade(self, prompt: str) -> bool:
        """Determine if a prompt should be upgraded."""
        if not self.enabled:
            return False

        prompt_lower = prompt.lower().strip()

        # Skip very short or greeting-like inputs
        if len(prompt_lower) < self.min_length:
            return False

        # Skip if already very detailed
        if len(prompt) > self.max_upgrade_length:
            return False

        # Skip common greetings/acknowledgments
        for pattern in SKIP_PATTERNS:
            if prompt_lower == pattern or prompt_lower.startswith(pattern + " "):
                return False

        return True

    async def upgrade(self, prompt: str) -> str:
        """
        Upgrade a user prompt to professional quality.
        Returns original prompt if upgrade fails or not needed.
        """
        if not self.should_upgrade(prompt):
            return prompt

        try:
            # Try Grok first (fast)
            grok = self._get_grok()
            if grok:
                upgraded = await self._upgrade_with_grok(prompt)
                if upgraded:
                    logger.info(f"Prompt upgraded by Grok: '{prompt[:30]}...' -> '{upgraded[:50]}...'")
                    return upgraded

            # Fallback to Gemini
            gemini = self._get_gemini()
            if gemini:
                upgraded = await self._upgrade_with_gemini(prompt)
                if upgraded:
                    logger.info(f"Prompt upgraded by Gemini: '{prompt[:30]}...' -> '{upgraded[:50]}...'")
                    return upgraded

        except Exception as e:
            logger.warning(f"Prompt upgrade failed: {e}")

        return prompt

    async def _upgrade_with_grok(self, prompt: str) -> Optional[str]:
        """Use Grok to upgrade the prompt."""
        grok = self._get_grok()
        if not grok:
            return None

        try:
            response = await grok.chat(
                message=f"Upgrade this prompt:\n\n{prompt}",
                system_prompt=UPGRADER_SYSTEM_PROMPT,
                max_tokens=300
            )

            if response and response.get("content"):
                upgraded = response["content"].strip()
                # Sanity check - don't return if it's way longer or seems wrong
                if len(upgraded) < len(prompt) * 5 and len(upgraded) > 0:
                    return upgraded

        except Exception as e:
            logger.debug(f"Grok upgrade failed: {e}")

        return None

    async def _upgrade_with_gemini(self, prompt: str) -> Optional[str]:
        """Use Gemini to upgrade the prompt."""
        gemini = self._get_gemini()
        if not gemini:
            return None

        try:
            response = await gemini.chat(
                message=f"Upgrade this prompt:\n\n{prompt}",
                system_prompt=UPGRADER_SYSTEM_PROMPT,
                max_tokens=300
            )

            if response and response.get("content"):
                upgraded = response["content"].strip()
                if len(upgraded) < len(prompt) * 5 and len(upgraded) > 0:
                    return upgraded

        except Exception as e:
            logger.debug(f"Gemini upgrade failed: {e}")

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get upgrader status."""
        return {
            "enabled": self.enabled,
            "grok_available": self._get_grok() is not None,
            "gemini_available": self._get_gemini() is not None,
            "min_length": self.min_length,
            "max_upgrade_length": self.max_upgrade_length
        }


# Global singleton
_prompt_upgrader = None

def get_prompt_upgrader() -> PromptUpgrader:
    """Get the global prompt upgrader instance."""
    global _prompt_upgrader
    if _prompt_upgrader is None:
        _prompt_upgrader = PromptUpgrader()
    return _prompt_upgrader


async def upgrade_prompt(prompt: str) -> str:
    """Convenience function to upgrade a prompt."""
    upgrader = get_prompt_upgrader()
    return await upgrader.upgrade(prompt)
