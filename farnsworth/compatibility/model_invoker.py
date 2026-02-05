"""
Farnsworth Model Invoker - Unified Model Calling Layer
=======================================================

Connects task routing to actual model invocation.

This is the bridge between:
- OpenClaw compatibility layer (tools/skills)
- Task routing (which model to use)
- Actual model APIs (Grok, Gemini, Claude, etc.)

Flow:
1. Tool call comes in → classify task type
2. Task routing selects best model
3. Model invoker calls the actual API
4. Fallback chain if model fails
5. Return result

"The right model, called the right way." - The Collective
"""

import asyncio
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from .task_routing import (
    OpenClawTaskType,
    classify_openclaw_tool,
    get_best_model_for_task,
    get_fallback_chain,
    get_model_for_channel,
    MODEL_REGISTRY,
)


@dataclass
class ModelResponse:
    """Response from a model invocation."""
    success: bool
    model_id: str
    content: str = ""
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0
    fallback_used: bool = False
    fallback_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelInvoker:
    """
    Unified model invocation layer.

    Handles:
    - Model selection via task routing
    - Actual API calls to each provider
    - Fallback chains on failure
    - Response normalization
    """

    def __init__(self):
        self._initialized = False
        self._providers: Dict[str, Any] = {}
        self._shadow_agents_available = False

    async def initialize(self):
        """Initialize model providers."""
        if self._initialized:
            return

        # Load available providers
        await self._load_providers()
        self._initialized = True
        logger.info(f"ModelInvoker initialized with {len(self._providers)} providers")

    async def _load_providers(self):
        """Load all available model providers."""

        # Grok (xAI)
        try:
            from farnsworth.integration.external.grok import get_grok_provider
            provider = get_grok_provider()
            if provider and provider.api_key:
                self._providers["Grok"] = provider
                logger.debug("Loaded Grok provider")
        except Exception as e:
            logger.debug(f"Grok not available: {e}")

        # Gemini (Google)
        try:
            from farnsworth.integration.external.gemini import get_gemini_provider
            provider = get_gemini_provider()
            if provider:
                self._providers["Gemini"] = provider
                logger.debug("Loaded Gemini provider")
        except Exception as e:
            logger.debug(f"Gemini not available: {e}")

        # Claude (Anthropic)
        try:
            from farnsworth.integration.external.claude import get_claude_provider
            provider = get_claude_provider()
            if provider:
                self._providers["Claude"] = provider
                self._providers["ClaudeOpus"] = provider  # Same provider, different model
                logger.debug("Loaded Claude provider")
        except Exception as e:
            logger.debug(f"Claude not available: {e}")

        # Kimi (Moonshot)
        try:
            from farnsworth.integration.external.kimi import get_kimi_provider
            provider = get_kimi_provider()
            if provider:
                self._providers["Kimi"] = provider
                logger.debug("Loaded Kimi provider")
        except Exception as e:
            logger.debug(f"Kimi not available: {e}")

        # DeepSeek
        try:
            from farnsworth.integration.external.base import get_provider
            provider = get_provider("deepseek")
            if provider:
                self._providers["DeepSeek"] = provider
                logger.debug("Loaded DeepSeek provider")
        except Exception as e:
            logger.debug(f"DeepSeek not available: {e}")

        # HuggingFace (Local)
        try:
            from farnsworth.integration.external.huggingface import get_huggingface_provider
            provider = get_huggingface_provider()
            if provider:
                self._providers["HuggingFace"] = provider
                logger.debug("Loaded HuggingFace provider")
        except Exception as e:
            logger.debug(f"HuggingFace not available: {e}")

        # Shadow agents (tmux persistent)
        try:
            from farnsworth.core.collective.persistent_agent import call_shadow_agent, AGENT_CONFIGS
            if AGENT_CONFIGS:
                self._shadow_agents_available = True
                logger.debug("Shadow agents available")
        except Exception as e:
            logger.debug(f"Shadow agents not available: {e}")

    async def invoke(
        self,
        prompt: str,
        task_type: OpenClawTaskType = None,
        tool: str = None,
        action: str = None,
        preferred_model: str = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        use_fallback: bool = True,
        **kwargs
    ) -> ModelResponse:
        """
        Invoke the best model for a task.

        Args:
            prompt: The prompt to send
            task_type: Explicit task type (or auto-detect from tool/action)
            tool: OpenClaw tool name (for auto-classification)
            action: OpenClaw action (for auto-classification)
            preferred_model: Override model selection
            max_tokens: Max response tokens
            temperature: Sampling temperature
            use_fallback: Try fallback models on failure
            **kwargs: Additional model-specific params

        Returns:
            ModelResponse with result
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Determine task type
        if task_type is None and tool:
            task_type = classify_openclaw_tool(tool, action)
        elif task_type is None:
            task_type = OpenClawTaskType.RUNTIME  # Default

        # Select model
        if preferred_model and preferred_model in self._providers:
            model_id = preferred_model
        else:
            model_id = get_best_model_for_task(
                task_type,
                exclude_models=[m for m in MODEL_REGISTRY if m not in self._providers]
            )

        if not model_id:
            return ModelResponse(
                success=False,
                model_id="none",
                error="No models available for this task type"
            )

        # Try primary model
        result = await self._call_model(model_id, prompt, max_tokens, temperature, **kwargs)

        if result.success:
            result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        # Try fallback chain
        if use_fallback:
            fallback_chain = get_fallback_chain(model_id, task_type)
            result.fallback_chain = [model_id]

            for fallback_model in fallback_chain:
                if fallback_model not in self._providers:
                    continue

                logger.info(f"Trying fallback model: {fallback_model}")
                result.fallback_chain.append(fallback_model)

                fallback_result = await self._call_model(
                    fallback_model, prompt, max_tokens, temperature, **kwargs
                )

                if fallback_result.success:
                    fallback_result.fallback_used = True
                    fallback_result.fallback_chain = result.fallback_chain
                    fallback_result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return fallback_result

        result.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    async def _call_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> ModelResponse:
        """Call a specific model."""
        try:
            provider = self._providers.get(model_id)
            if not provider:
                # Try shadow agent
                if self._shadow_agents_available:
                    return await self._call_shadow_agent(model_id, prompt, max_tokens)
                return ModelResponse(
                    success=False,
                    model_id=model_id,
                    error=f"Provider not available: {model_id}"
                )

            # Call based on provider type
            if model_id == "Grok":
                result = await self._call_grok(provider, prompt, max_tokens, temperature, **kwargs)
            elif model_id == "Gemini":
                result = await self._call_gemini(provider, prompt, max_tokens, temperature, **kwargs)
            elif model_id in ("Claude", "ClaudeOpus"):
                result = await self._call_claude(provider, prompt, max_tokens, temperature, model_id, **kwargs)
            elif model_id == "Kimi":
                result = await self._call_kimi(provider, prompt, max_tokens, temperature, **kwargs)
            elif model_id == "DeepSeek":
                result = await self._call_deepseek(provider, prompt, max_tokens, temperature, **kwargs)
            elif model_id == "HuggingFace":
                result = await self._call_huggingface(provider, prompt, max_tokens, temperature, **kwargs)
            else:
                # Generic call
                result = await self._call_generic(provider, prompt, max_tokens, temperature, **kwargs)

            return result

        except Exception as e:
            logger.error(f"Model call failed ({model_id}): {e}")
            return ModelResponse(
                success=False,
                model_id=model_id,
                error=str(e)
            )

    async def _call_grok(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Call Grok API."""
        try:
            result = await provider.chat(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            if result and result.get("content"):
                return ModelResponse(
                    success=True,
                    model_id="Grok",
                    content=result["content"],
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"model": result.get("model", "grok-4")}
                )
            return ModelResponse(success=False, model_id="Grok", error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id="Grok", error=str(e))

    async def _call_gemini(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Call Gemini API."""
        try:
            result = await provider.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            if result and result.get("text"):
                return ModelResponse(
                    success=True,
                    model_id="Gemini",
                    content=result["text"],
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"model": result.get("model", "gemini-3-flash")}
                )
            return ModelResponse(success=False, model_id="Gemini", error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id="Gemini", error=str(e))

    async def _call_claude(self, provider, prompt: str, max_tokens: int, temperature: float, model_variant: str, **kwargs) -> ModelResponse:
        """Call Claude API."""
        try:
            model = "claude-sonnet-4-5-20250514" if model_variant == "Claude" else "claude-opus-4-5-20250514"
            result = await provider.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                **kwargs
            )
            if result and result.get("content"):
                return ModelResponse(
                    success=True,
                    model_id=model_variant,
                    content=result["content"],
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"model": model}
                )
            return ModelResponse(success=False, model_id=model_variant, error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id=model_variant, error=str(e))

    async def _call_kimi(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Call Kimi API."""
        try:
            result = await provider.chat(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            if result and result.get("content"):
                return ModelResponse(
                    success=True,
                    model_id="Kimi",
                    content=result["content"],
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"model": "kimi-k2.5"}
                )
            return ModelResponse(success=False, model_id="Kimi", error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id="Kimi", error=str(e))

    async def _call_deepseek(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Call DeepSeek API."""
        try:
            # Use thinking mode for complex tasks
            use_thinking = kwargs.pop("thinking_mode", False)
            endpoint = "deepseek-reasoner" if use_thinking else "deepseek-chat"

            result = await provider.chat(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=endpoint,
                **kwargs
            )
            if result and result.get("content"):
                return ModelResponse(
                    success=True,
                    model_id="DeepSeek",
                    content=result["content"],
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    metadata={"model": endpoint, "thinking": use_thinking}
                )
            return ModelResponse(success=False, model_id="DeepSeek", error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id="DeepSeek", error=str(e))

    async def _call_huggingface(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Call HuggingFace local models."""
        try:
            model_name = kwargs.pop("model", "mistral-7b")
            result = await provider.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model_name,
                **kwargs
            )
            if result and result.get("text"):
                return ModelResponse(
                    success=True,
                    model_id="HuggingFace",
                    content=result["text"],
                    metadata={"model": model_name, "local": True}
                )
            return ModelResponse(success=False, model_id="HuggingFace", error="Empty response")
        except Exception as e:
            return ModelResponse(success=False, model_id="HuggingFace", error=str(e))

    async def _call_generic(self, provider, prompt: str, max_tokens: int, temperature: float, **kwargs) -> ModelResponse:
        """Generic provider call."""
        try:
            # Try common method names
            for method_name in ["chat", "generate", "complete", "invoke"]:
                method = getattr(provider, method_name, None)
                if method and callable(method):
                    result = await method(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                    if result:
                        content = result.get("content") or result.get("text") or str(result)
                        return ModelResponse(
                            success=True,
                            model_id="generic",
                            content=content
                        )
            return ModelResponse(success=False, model_id="generic", error="No callable method found")
        except Exception as e:
            return ModelResponse(success=False, model_id="generic", error=str(e))

    async def _call_shadow_agent(self, agent_id: str, prompt: str, max_tokens: int) -> ModelResponse:
        """Call a tmux shadow agent."""
        try:
            from farnsworth.core.collective.persistent_agent import call_shadow_agent

            # Map model IDs to shadow agent names
            shadow_map = {
                "Grok": "grok",
                "Gemini": "gemini",
                "Claude": "claude",
                "Kimi": "kimi",
                "DeepSeek": "deepseek",
                "Phi": "phi",
                "HuggingFace": "huggingface",
            }

            shadow_name = shadow_map.get(agent_id, agent_id.lower())
            result = await call_shadow_agent(shadow_name, prompt, max_tokens)

            if result and result[1]:
                return ModelResponse(
                    success=True,
                    model_id=agent_id,
                    content=result[1],
                    metadata={"shadow_agent": shadow_name}
                )
            return ModelResponse(success=False, model_id=agent_id, error="Shadow agent returned empty")
        except Exception as e:
            return ModelResponse(success=False, model_id=agent_id, error=str(e))

    async def invoke_for_tool(
        self,
        tool: str,
        action: str,
        prompt: str,
        context: Dict = None,
        **kwargs
    ) -> ModelResponse:
        """
        Convenience method to invoke model for an OpenClaw tool.

        Args:
            tool: OpenClaw tool name
            action: Tool action
            prompt: Prompt with tool context
            context: Additional context dict
            **kwargs: Model params

        Returns:
            ModelResponse
        """
        task_type = classify_openclaw_tool(tool, action)

        # Add tool context to prompt
        full_prompt = f"""Tool: {tool}
Action: {action}
Context: {context or {}}

Task: {prompt}"""

        return await self.invoke(
            full_prompt,
            task_type=task_type,
            tool=tool,
            action=action,
            **kwargs
        )

    async def invoke_for_channel(
        self,
        channel_type: str,
        message: str,
        context: Dict = None,
        **kwargs
    ) -> ModelResponse:
        """
        Invoke model for a messaging channel.

        Args:
            channel_type: Channel type (telegram, discord, etc.)
            message: User message
            context: Conversation context
            **kwargs: Model params

        Returns:
            ModelResponse
        """
        preferred_model = get_model_for_channel(channel_type)

        return await self.invoke(
            message,
            task_type=OpenClawTaskType.MESSAGING,
            preferred_model=preferred_model,
            **kwargs
        )

    def get_available_models(self) -> List[str]:
        """Get list of available model IDs."""
        models = list(self._providers.keys())
        if self._shadow_agents_available:
            models.extend(["Phi"])  # Add shadow-only models
        return models

    def get_status(self) -> Dict[str, Any]:
        """Get invoker status."""
        return {
            "initialized": self._initialized,
            "providers_loaded": len(self._providers),
            "available_models": self.get_available_models(),
            "shadow_agents": self._shadow_agents_available,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_model_invoker: Optional[ModelInvoker] = None


def get_model_invoker() -> ModelInvoker:
    """Get or create the global model invoker."""
    global _model_invoker
    if _model_invoker is None:
        _model_invoker = ModelInvoker()
    return _model_invoker


async def invoke_model(
    prompt: str,
    task_type: OpenClawTaskType = None,
    tool: str = None,
    action: str = None,
    **kwargs
) -> ModelResponse:
    """
    Convenience function to invoke a model.

    Auto-selects the best model based on task type.
    """
    invoker = get_model_invoker()
    return await invoker.invoke(prompt, task_type=task_type, tool=tool, action=action, **kwargs)


async def invoke_for_tool(tool: str, action: str, prompt: str, **kwargs) -> ModelResponse:
    """Invoke model for an OpenClaw tool."""
    invoker = get_model_invoker()
    return await invoker.invoke_for_tool(tool, action, prompt, **kwargs)


async def invoke_for_channel(channel_type: str, message: str, **kwargs) -> ModelResponse:
    """Invoke model for a messaging channel."""
    invoker = get_model_invoker()
    return await invoker.invoke_for_channel(channel_type, message, **kwargs)
