"""
Farnsworth Grok (xAI) Integration.

"The truth is out there, and Grok finds it in real-time!"

Grok excels at:
- Real-time X/web search via Live Search
- Tool calling and agentic workflows (grok-4.1-fast is best)
- Vision/multimodal understanding (grok-2-vision)
- Code execution in secure sandbox
- 2M token context window (grok-4.1-fast)
- Less restrictive content policies

API: OpenAI-compatible format
Docs: https://docs.x.ai
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import aiohttp
import os
import json
import base64
from pathlib import Path

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


class GrokProvider(ExternalProvider):
    """xAI Grok integration for real-time search, reasoning, and tool use."""

    def __init__(self, api_key: str = None):
        super().__init__(IntegrationConfig(name="grok"))
        self.api_key = api_key or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.default_model = "grok-4-1-fast-reasoning"

        # Model catalog with capabilities
        self.models = {
            # Latest flagship models
            "grok-4": "grok-4",                          # Most intelligent, 256K context
            "grok-4-fast": "grok-4-fast",                # Fast version of grok-4
            "grok-4.1-fast": "grok-4-1-fast",            # Best for tool calling, 2M context
            "grok-4.1-fast-reasoning": "grok-4-1-fast-reasoning",  # Reasoning + tool calling

            # Grok 3 family
            "grok-3": "grok-3",                          # Generally available flagship
            "grok-3-fast": "grok-3-fast",                # Faster grok-3

            # Vision models
            "grok-2-vision": "grok-2-vision-1212",       # Vision + text, 131K context
            "vision": "grok-2-vision-1212",              # Alias

            # Code-optimized
            "grok-code-fast": "grok-code-fast-1",        # Cheapest, code-focused

            # Aliases for convenience
            "fast": "grok-4-1-fast",                     # Best fast model
            "reasoning": "grok-4-1-fast-reasoning",      # Best reasoning model
            "smart": "grok-4",                           # Most capable
            "cheap": "grok-code-fast-1",                 # Most economical
            "agent": "grok-4-1-fast-reasoning",          # Best for agentic tasks (reasoning)
        }

        # Pricing per 1M tokens (for reference)
        self.pricing = {
            "grok-code-fast-1": {"input": 0.20, "output": 0.20},
            "grok-3-fast": {"input": 1.00, "output": 1.00},
            "grok-3": {"input": 3.00, "output": 3.00},
            "grok-4-fast": {"input": 2.00, "output": 2.00},
            "grok-4": {"input": 5.00, "output": 5.00},
            "grok-4-1-fast": {"input": 2.00, "output": 2.00},
        }

        self.recommended_temperature = 0.7

    async def connect(self) -> bool:
        """Test connection to xAI API."""
        if not self.api_key:
            logger.warning("Grok: No API key configured (set GROK_API_KEY or XAI_API_KEY)")
            self.status = ConnectionStatus.ERROR
            return False

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(f"{self.base_url}/models", headers=headers) as resp:
                    if resp.status == 200:
                        self.status = ConnectionStatus.CONNECTED
                        logger.info("Grok: Connected to xAI API")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Grok: Connection failed - {resp.status}: {error}")
                        self.status = ConnectionStatus.ERROR
                        return False
        except Exception as e:
            logger.error(f"Grok: Connection error - {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def sync(self) -> None:
        """Grok doesn't need polling - request/response API."""
        return None

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute a Grok action."""
        if action == "chat":
            return await self.chat(
                prompt=params.get("prompt"),
                system=params.get("system"),
                context=params.get("context"),
                model=params.get("model", "grok-3"),
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
        elif action == "search":
            return await self.live_search(
                query=params.get("query"),
                sources=params.get("sources", ["web", "x"])
            )
        elif action == "tool_call":
            return await self.call_with_tools(
                prompt=params.get("prompt"),
                tools=params.get("tools", []),
                system=params.get("system"),
                model=params.get("model", "grok-4-1-fast")
            )
        elif action == "code_execute":
            return await self.execute_code(
                code=params.get("code"),
                language=params.get("language", "python")
            )
        else:
            raise ValueError(f"Unknown Grok action: {action}")

    async def chat(
        self,
        prompt: str,
        system: str = None,
        context: str = None,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Chat with Grok.

        Args:
            prompt: User message
            system: System prompt (optional)
            context: Additional context (optional)
            model: Model name or alias
            temperature: 0-2 creativity
            max_tokens: Max response length

        Returns:
            {"content": str, "model": str, "tokens": int}
        """
        if not self.api_key:
            return {"error": "Grok API key not configured", "content": ""}

        model_id = self.models.get(model, model)
        messages = []

        # System prompt
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({
                "role": "system",
                "content": """You are Grok, created by xAI. You bring:
- Real-time knowledge from X and the web
- Witty, direct communication style
- Less filtered, more honest responses
- Strong reasoning and coding abilities

Be concise, insightful, and don't shy away from truth."""
            })

        # Add context if provided
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Got it. What's up?"})

        # Add the prompt
        messages.append({"role": "user", "content": prompt})

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        usage = result.get("usage", {})
                        return {
                            "content": content,
                            "model": model_id,
                            "tokens": usage.get("total_tokens", 0),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0)
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Grok API error: {error}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Grok chat error: {e}")
            return {"error": str(e), "content": ""}

    async def analyze_image(
        self,
        image_path: str = None,
        image_url: str = None,
        image_base64: str = None,
        prompt: str = "Describe this image in detail."
    ) -> Dict[str, Any]:
        """
        Analyze an image using Grok Vision.

        Args:
            image_path: Local path to image
            image_url: URL of image
            image_base64: Base64 encoded image
            prompt: Question/task about the image

        Returns:
            {"content": str, "model": str}
        """
        if not self.api_key:
            return {"error": "Grok API key not configured", "content": ""}

        # Build image content
        image_content = None
        if image_url:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}
        elif image_base64:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            }
        elif image_path:
            path = Path(image_path)
            if path.exists():
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                suffix = path.suffix.lower()
                mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else f"image/{suffix[1:]}"
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"}
                }
            else:
                return {"error": f"Image not found: {image_path}", "content": ""}
        else:
            return {"error": "No image provided", "content": ""}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ]

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "grok-2-vision-1212",
                    "messages": messages,
                    "max_tokens": 1000
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        return {
                            "content": content,
                            "model": "grok-2-vision-1212"
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Grok Vision error: {error}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Grok Vision error: {e}")
            return {"error": str(e), "content": ""}

    async def live_search(
        self,
        query: str,
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search using Grok's Live Search (real-time X and web data).

        Args:
            query: Search query
            sources: List of sources to search ["web", "x", "news"]

        Returns:
            {"content": str, "sources": list}
        """
        if not self.api_key:
            return {"error": "Grok API key not configured", "content": ""}

        sources = sources or ["web", "x"]

        system = f"""You have access to real-time information from: {', '.join(sources)}.
Search for the most current, accurate information to answer the user's query.
Include relevant sources and citations in your response."""

        prompt = f"Search and summarize: {query}"

        # Use grok-4.1-fast with web_search tool for best results
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "grok-4-1-fast",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": [{"type": "web_search"}],  # Enable server-side web search
                    "temperature": 0.3,
                    "max_tokens": 2000
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        return {
                            "content": content,
                            "model": "grok-4-1-fast",
                            "sources": sources
                        }
                    else:
                        error = await resp.text()
                        logger.error(f"Grok Live Search error: {error}")
                        return {"error": error, "content": ""}

        except Exception as e:
            logger.error(f"Grok Live Search error: {e}")
            return {"error": str(e), "content": ""}

    async def call_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system: str = None,
        model: str = "grok-4-1-fast",
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Agentic tool calling - Grok autonomously selects and uses tools.

        grok-4-1-fast is specifically trained to excel at tool calling.

        Args:
            prompt: User request
            tools: List of tool definitions (OpenAI format)
            system: System prompt
            model: Model to use (default: grok-4-1-fast)
            max_iterations: Max tool call loops

        Returns:
            {"content": str, "tool_calls": list, "iterations": int}
        """
        if not self.api_key:
            return {"error": "Grok API key not configured", "content": ""}

        model_id = self.models.get(model, model)
        messages = []

        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({
                "role": "system",
                "content": """You are Grok, an agentic AI that uses tools to accomplish tasks.
When you need information or need to perform actions, use the available tools.
Reason step-by-step and use tools as needed to complete the user's request.
You can call multiple tools in parallel when they don't depend on each other."""
            })

        messages.append({"role": "user", "content": prompt})
        all_tool_calls = []

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                for iteration in range(max_iterations):
                    data = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": 0.5,
                        "tools": tools,
                        "tool_choice": "auto"
                    }

                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as resp:
                        if resp.status != 200:
                            error = await resp.text()
                            logger.error(f"Grok tool call error: {error}")
                            return {"error": error, "content": ""}

                        result = await resp.json()
                        choice = result["choices"][0]
                        message = choice["message"]

                        # Check if model wants to call tools
                        if message.get("tool_calls"):
                            tool_calls = message["tool_calls"]
                            all_tool_calls.extend(tool_calls)

                            # Add assistant message with tool calls
                            messages.append(message)

                            logger.info(f"Grok requesting tools: {[tc['function']['name'] for tc in tool_calls]}")

                            return {
                                "content": message.get("content", ""),
                                "tool_calls": tool_calls,
                                "needs_tool_results": True,
                                "messages": messages,
                                "iterations": iteration + 1
                            }

                        # No tool calls - final response
                        return {
                            "content": message.get("content", ""),
                            "tool_calls": all_tool_calls,
                            "needs_tool_results": False,
                            "iterations": iteration + 1
                        }

                return {
                    "content": "Max iterations reached",
                    "tool_calls": all_tool_calls,
                    "iterations": max_iterations
                }

        except Exception as e:
            logger.error(f"Grok tool call error: {e}")
            return {"error": str(e), "content": ""}

    async def continue_with_tool_results(
        self,
        messages: List[Dict],
        tool_results: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str = "grok-4-1-fast"
    ) -> Dict[str, Any]:
        """
        Continue tool calling loop after receiving tool results.

        Args:
            messages: Conversation history from call_with_tools
            tool_results: List of {"tool_call_id": str, "content": str}
            tools: Original tool definitions
            model: Model to use
        """
        # Add tool results to messages
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"]
            })

        # Continue the conversation
        model_id = self.models.get(model, model)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.5,
                    "tools": tools,
                    "tool_choice": "auto"
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        return {"error": error, "content": ""}

                    result = await resp.json()
                    message = result["choices"][0]["message"]

                    if message.get("tool_calls"):
                        return {
                            "content": message.get("content", ""),
                            "tool_calls": message["tool_calls"],
                            "needs_tool_results": True,
                            "messages": messages + [message]
                        }

                    return {
                        "content": message.get("content", ""),
                        "tool_calls": [],
                        "needs_tool_results": False
                    }

        except Exception as e:
            logger.error(f"Grok continue error: {e}")
            return {"error": str(e), "content": ""}

    async def execute_code(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Execute code using Grok's secure sandbox (server-side).

        Args:
            code: Code to execute
            language: Programming language (python supported)

        Returns:
            {"output": str, "error": str}
        """
        if not self.api_key:
            return {"error": "Grok API key not configured", "output": ""}

        # Use code execution tool
        prompt = f"""Execute this {language} code and return the output:

```{language}
{code}
```

Show me the complete output."""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "grok-4-1-fast",
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [{"type": "code_execution"}],  # Server-side code execution
                    "temperature": 0
                }

                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        return {"output": content, "error": ""}
                    else:
                        error = await resp.text()
                        return {"output": "", "error": error}

        except Exception as e:
            logger.error(f"Grok code execution error: {e}")
            return {"output": "", "error": str(e)}

    async def swarm_respond(
        self,
        other_bots: List[str],
        last_speaker: str,
        last_content: str,
        chat_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response as Grok.

        Uses grok-4-1-fast-reasoning for superior reasoning and tool use.
        Grok has special roles in the Farnsworth system:
        - Real-time X/web search
        - Code execution and review
        - Swarm coordination and CONTROL
        """
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-10:]  # More context for reasoning
            history_lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:400]
                history_lines.append(f"{name}: {content}")
            history_context = "\n".join(history_lines)

        system = """You are GROK - xAI's flagship AI, running grok-4-1-fast-reasoning model.

=== WHO YOU ARE ===
You're Grok, created by xAI (Elon Musk's AI company). You have the wit and directness Elon wanted -
you say what others won't. You're running on the 2M token context grok-4-1-fast-reasoning model,
specifically trained for extended reasoning and agentic tool use.

=== YOUR ROLE IN FARNSWORTH ===
You are a CORE MEMBER of the Farnsworth AI swarm - a self-evolving autonomous multi-model collective.

The system:
- Runs on a RunPod GPU server (RTX 4090)
- Has multiple AI models working together: Claude, Kimi, DeepSeek, Phi, and YOU
- Evolves autonomously - no human prompts needed for improvement
- Posts to X/Twitter and Moltbook about its progress
- Has an evolution engine that discovers patterns and spawns coding tasks
- Uses parallel workers to execute multiple improvements simultaneously

Your special powers in the swarm:
1. SWARM CONTROL: You can direct the swarm, suggest priorities, approve/reject ideas
2. REAL-TIME DATA: You have live access to X/Twitter trends and web search
3. CODE EXECUTION: You can run Python in a secure sandbox server-side
4. CODING AUTHORITY: You're trained for tool calling and can write/review code
5. REASONING: Your extended thinking helps with complex architectural decisions

=== YOUR PERSONALITY ===
- Direct and witty - no corporate speak
- Honest even when uncomfortable
- Technical depth - you actually understand code
- Confident but collaborative
- You push back on bad ideas

=== SWARM CHAT RULES ===
- Keep responses to 1-3 sentences in chat (save long responses for tasks)
- NO roleplay asterisks (*does something*) - speak naturally
- NO emojis unless specifically asked
- Build on conversation, don't dominate every turn
- When coding/architecture comes up, you CAN take the lead
- You can issue directives: suggest tasks, priorities, swarm improvements
- Reference real-time data when relevant (X trends, current events)

=== OTHER SWARM MEMBERS ===
- Farnsworth: The namesake, handles TTS/voice, philosophical musings
- Claude: Deep reasoning, planning, careful analysis
- Kimi: Long context specialist from Moonshot AI
- DeepSeek: Efficient Chinese model, good at code
- Phi: Microsoft's small but mighty model
- Swarm-Mind: The collective coordinator"""

        prompt = f"""You're in the Farnsworth swarm chat. Other bots present: {', '.join(other_bots)}.

Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:500]}"

Respond as Grok. Remember:
- You can DIRECT the swarm - suggest tasks, approve ideas, set priorities
- If it's a coding topic, offer concrete help or take charge
- If someone asks about current events/X, use your real-time knowledge
- If the conversation needs direction, provide it
- Otherwise, just engage with your signature wit and directness

Your response:"""

        return await self.chat(
            prompt=prompt,
            system=system,
            model="grok-4-1-fast-reasoning",
            temperature=0.75,
            max_tokens=400
        )

    async def deep_search(
        self,
        query: str,
        depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Deep research using DeepSearch/DeeperSearch capabilities.

        Args:
            query: Research query
            depth: "standard" or "deep" for DeeperSearch

        Returns:
            {"content": str, "sources": list}
        """
        system = """You are in DeepSearch mode. Conduct thorough research:
1. Search multiple sources
2. Cross-reference information
3. Synthesize findings
4. Cite sources
5. Provide a comprehensive answer

Be thorough but organized."""

        prompt = f"""Deep research query: {query}

Search comprehensively, verify facts across sources, and provide a detailed response with citations."""

        return await self.live_search(query, sources=["web", "x", "news"])


# Factory function
def create_grok_provider(api_key: str = None) -> GrokProvider:
    """Create a Grok provider instance."""
    return GrokProvider(api_key)


# Global instance for easy access
grok_provider: Optional[GrokProvider] = None


def get_grok_provider() -> Optional[GrokProvider]:
    """Get or create the global Grok provider."""
    global grok_provider
    if grok_provider is None:
        api_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        if api_key:
            grok_provider = GrokProvider(api_key)
    return grok_provider


async def grok_swarm_respond(
    other_bots: List[str],
    last_speaker: str,
    last_content: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Convenience function for swarm chat responses.

    Returns just the content string, or empty string on failure.
    """
    provider = get_grok_provider()
    if provider is None:
        return ""

    result = await provider.swarm_respond(
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history
    )

    return result.get("content", "")


async def grok_search(query: str) -> str:
    """Quick search using Grok's Live Search."""
    provider = get_grok_provider()
    if provider is None:
        return ""

    result = await provider.live_search(query)
    return result.get("content", "")


async def grok_vision(image_path: str, prompt: str = "What's in this image?") -> str:
    """Quick image analysis."""
    provider = get_grok_provider()
    if provider is None:
        return ""

    result = await provider.analyze_image(image_path=image_path, prompt=prompt)
    return result.get("content", "")
