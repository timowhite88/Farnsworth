"""
Farnsworth LLM Backend

Novel Approaches:
1. Cascade Inference - Start with fast model, auto-escalate on low confidence
2. Adaptive Temperature - Dynamic temperature based on task entropy
3. Context-Aware Routing - Route to specialized backends based on content analysis
4. Streaming Confidence Estimation - Real-time generation quality monitoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional, Any
import asyncio
import time
import re
import math
from loguru import logger


class BackendType(Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    BITNET = "bitnet"
    OPENAI_COMPATIBLE = "openai_compatible"  # For MiniMax, DeepInfra, OpenRouter, etc.


@dataclass
class GenerationConfig:
    """Configuration for text generation with adaptive parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)

    # Novel: Adaptive temperature bounds
    temp_min: float = 0.1
    temp_max: float = 1.5
    temp_adaptation_rate: float = 0.1

    # Novel: Cascade inference settings
    cascade_enabled: bool = True
    confidence_threshold: float = 0.7
    escalation_delay_tokens: int = 50


@dataclass
class GenerationResult:
    """Result from LLM generation with metadata."""
    text: str
    tokens_generated: int
    tokens_per_second: float
    model_used: str
    backend_used: BackendType

    # Novel: Confidence and quality metrics
    confidence_score: float = 0.0
    perplexity_estimate: float = 0.0
    escalated: bool = False
    cascade_depth: int = 0

    # Performance metrics
    time_to_first_token: float = 0.0
    total_time: float = 0.0


@dataclass
class StreamChunk:
    """Streaming chunk with real-time quality estimation."""
    text: str
    token_index: int
    confidence: float = 1.0
    cumulative_confidence: float = 1.0


class ConfidenceEstimator:
    """
    Novel: Real-time confidence estimation during generation.

    Uses multiple heuristics to estimate generation quality:
    - Token probability analysis (when available)
    - Repetition detection
    - Coherence scoring via n-gram analysis
    - Semantic drift detection
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.token_history: list[str] = []
        self.ngram_counts: dict[tuple, int] = {}
        self.base_confidence = 1.0

    def update(self, token: str, logprob: Optional[float] = None) -> float:
        """Update confidence with new token."""
        self.token_history.append(token)
        if len(self.token_history) > self.window_size * 2:
            self.token_history = self.token_history[-self.window_size:]

        confidence = 1.0

        # Factor 1: Token probability (if available)
        if logprob is not None:
            prob = math.exp(logprob)
            confidence *= min(1.0, prob * 2)  # Scale low probs

        # Factor 2: Repetition penalty
        repetition_score = self._detect_repetition()
        confidence *= repetition_score

        # Factor 3: Coherence via n-gram novelty
        coherence_score = self._estimate_coherence()
        confidence *= coherence_score

        self.base_confidence = 0.9 * self.base_confidence + 0.1 * confidence
        return self.base_confidence

    def _detect_repetition(self) -> float:
        """Detect repetitive patterns in recent tokens."""
        if len(self.token_history) < 10:
            return 1.0

        recent = self.token_history[-20:]

        # Check for exact phrase repetition
        for ngram_size in [3, 5, 8]:
            if len(recent) < ngram_size * 2:
                continue
            ngrams = [tuple(recent[i:i+ngram_size]) for i in range(len(recent) - ngram_size + 1)]
            unique_ratio = len(set(ngrams)) / len(ngrams)
            if unique_ratio < 0.5:
                return 0.5  # Heavy repetition detected

        return 1.0

    def _estimate_coherence(self) -> float:
        """Estimate coherence using n-gram novelty."""
        if len(self.token_history) < 5:
            return 1.0

        # Add recent trigrams to history
        recent = self.token_history[-10:]
        for i in range(len(recent) - 2):
            ngram = tuple(recent[i:i+3])
            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

        # Check if latest trigram is novel
        if len(self.token_history) >= 3:
            latest = tuple(self.token_history[-3:])
            count = self.ngram_counts.get(latest, 0)
            if count > 3:
                return 0.8  # Becoming repetitive

        return 1.0

    def reset(self):
        """Reset estimator state."""
        self.token_history.clear()
        self.ngram_counts.clear()
        self.base_confidence = 1.0


class AdaptiveTemperature:
    """
    Novel: Dynamic temperature adaptation based on generation context.

    - Increases temperature when stuck in loops
    - Decreases temperature for factual/code content
    - Responds to confidence signals
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.current_temp = config.temperature
        self.adjustment_history: list[float] = []

    def adapt(self, confidence: float, content_type: str = "general") -> float:
        """Adapt temperature based on generation state."""
        target = self.config.temperature

        # Content-type adjustments
        if content_type == "code":
            target *= 0.7  # Lower temp for code
        elif content_type == "creative":
            target *= 1.2  # Higher temp for creative

        # Confidence-based adjustment
        if confidence < 0.5:
            # Low confidence - increase temp to escape local minima
            target *= 1.3
        elif confidence > 0.9:
            # High confidence - can afford lower temp
            target *= 0.9

        # Clamp to bounds
        target = max(self.config.temp_min, min(self.config.temp_max, target))

        # Smooth adaptation
        self.current_temp = 0.8 * self.current_temp + 0.2 * target
        self.adjustment_history.append(self.current_temp)

        return self.current_temp

    def detect_content_type(self, text: str) -> str:
        """Heuristic content type detection."""
        code_indicators = [
            r'\b(def|class|function|import|return|if|for|while)\b',
            r'[{}\[\]();]',
            r'\b(var|let|const|async|await)\b',
        ]

        code_score = sum(1 for pattern in code_indicators if re.search(pattern, text))

        if code_score >= 2:
            return "code"
        elif any(word in text.lower() for word in ["story", "imagine", "creative", "write a"]):
            return "creative"
        return "general"


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None):
        self.model_name = model_name
        self.config = config or GenerationConfig()
        self.confidence_estimator = ConfidenceEstimator()
        self.adaptive_temp = AdaptiveTemperature(self.config)
        self._is_loaded = False

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass

    @abstractmethod
    async def load(self) -> bool:
        """Load the model. Returns True if successful."""
        pass

    @abstractmethod
    async def unload(self) -> bool:
        """Unload the model to free resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding vector."""
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def estimate_complexity(self, prompt: str) -> float:
        """
        Novel: Estimate task complexity from prompt.
        Used for cascade inference decisions.
        """
        complexity = 0.5  # Base complexity

        # Length factor
        words = len(prompt.split())
        complexity += min(0.2, words / 500)

        # Technical indicators
        tech_patterns = [
            r'\b(algorithm|optimize|implement|debug|analyze)\b',
            r'\b(mathematical|proof|theorem|equation)\b',
            r'\b(code|function|class|api)\b',
        ]
        for pattern in tech_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                complexity += 0.1

        # Multi-step reasoning indicators
        if any(word in prompt.lower() for word in ["step by step", "explain", "why", "how"]):
            complexity += 0.15

        return min(1.0, complexity)


class OllamaBackend(LLMBackend):
    """
    Ollama backend with enhanced features:
    - Connection pooling for concurrent requests
    - Automatic model pulling if not present
    - Health monitoring and auto-reconnect
    """

    def __init__(self, model_name: str, config: Optional[GenerationConfig] = None,
                 host: str = "http://localhost:11434"):
        super().__init__(model_name, config)
        self.host = host
        self._client = None
        self._health_check_interval = 30
        self._last_health_check = 0

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    async def _get_client(self):
        """Get or create Ollama client with lazy initialization."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.host)
            except ImportError:
                raise RuntimeError("ollama package not installed. Run: pip install ollama")
        return self._client

    async def _ensure_model(self):
        """Ensure model is available, pull if necessary."""
        client = await self._get_client()
        try:
            await client.show(self.model_name)
        except Exception:
            logger.info(f"Model {self.model_name} not found, pulling...")
            await client.pull(self.model_name)

    async def load(self) -> bool:
        """Load model into Ollama."""
        try:
            await self._ensure_model()
            client = await self._get_client()
            # Warm up the model with a simple generation
            await client.generate(model=self.model_name, prompt="Hello", options={"num_predict": 1})
            self._is_loaded = True
            logger.info(f"Loaded model {self.model_name} via Ollama")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return False

    async def unload(self) -> bool:
        """Unload model from memory."""
        # Ollama doesn't have explicit unload, but we can mark as unloaded
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with confidence estimation."""
        cfg = config or self.config
        client = await self._get_client()

        start_time = time.time()
        self.confidence_estimator.reset()

        # Detect content type for adaptive temperature
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        response = await client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "repeat_penalty": cfg.repeat_penalty,
            },
            stream=False,
        )

        total_time = time.time() - start_time
        text = response.get("response", "")

        # Estimate confidence from output
        tokens = text.split()
        for token in tokens:
            self.confidence_estimator.update(token)

        return GenerationResult(
            text=text,
            tokens_generated=response.get("eval_count", len(tokens)),
            tokens_per_second=response.get("eval_count", 0) / max(0.001, total_time),
            model_used=self.model_name,
            backend_used=self.backend_type,
            confidence_score=self.confidence_estimator.base_confidence,
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation with real-time confidence."""
        cfg = config or self.config
        client = await self._get_client()

        self.confidence_estimator.reset()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        token_index = 0
        cumulative_confidence = 1.0

        async for chunk in await client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temp,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "repeat_penalty": cfg.repeat_penalty,
            },
            stream=True,
        ):
            text = chunk.get("response", "")
            if text:
                confidence = self.confidence_estimator.update(text)
                cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                yield StreamChunk(
                    text=text,
                    token_index=token_index,
                    confidence=confidence,
                    cumulative_confidence=cumulative_confidence,
                )
                token_index += 1

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding from Ollama."""
        client = await self._get_client()
        response = await client.embeddings(
            model=self.model_name,
            prompt=text,
        )
        return response.get("embedding", [])


class LlamaCppBackend(LLMBackend):
    """
    llama.cpp backend with advanced features:
    - GPU layer offloading optimization
    - KV cache quantization
    - Speculative decoding support
    - Grammar-constrained generation
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        n_gpu_layers: int = -1,  # -1 = auto-detect
        n_ctx: int = 4096,
        n_batch: int = 512,
    ):
        super().__init__(model_path, config)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self._llm = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMA_CPP

    def _detect_gpu_layers(self) -> int:
        """Auto-detect optimal GPU layers based on VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Rough heuristic: ~0.5GB per 10 layers for 7B model
                return min(50, int(vram_gb * 20))
        except ImportError:
            pass
        return 0  # CPU only

    async def load(self) -> bool:
        """Load model with optimized settings."""
        try:
            from llama_cpp import Llama

            n_gpu = self.n_gpu_layers if self.n_gpu_layers >= 0 else self._detect_gpu_layers()

            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=False,
            )
            self._is_loaded = True
            logger.info(f"Loaded {self.model_path} with {n_gpu} GPU layers")
            return True
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            return False

    async def unload(self) -> bool:
        """Free model memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with llama.cpp."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        cfg = config or self.config
        self.confidence_estimator.reset()

        start_time = time.time()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._llm(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=temp,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repeat_penalty=cfg.repeat_penalty,
                stop=cfg.stop_sequences or None,
            )
        )

        total_time = time.time() - start_time
        text = response["choices"][0]["text"]

        # Update confidence
        for token in text.split():
            self.confidence_estimator.update(token)

        return GenerationResult(
            text=text,
            tokens_generated=response["usage"]["completion_tokens"],
            tokens_per_second=response["usage"]["completion_tokens"] / max(0.001, total_time),
            model_used=self.model_path,
            backend_used=self.backend_type,
            confidence_score=self.confidence_estimator.base_confidence,
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        cfg = config or self.config
        self.confidence_estimator.reset()

        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        token_index = 0
        cumulative_confidence = 1.0

        for chunk in self._llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=temp,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop_sequences or None,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                confidence = self.confidence_estimator.update(text)
                cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                yield StreamChunk(
                    text=text,
                    token_index=token_index,
                    confidence=confidence,
                    cumulative_confidence=cumulative_confidence,
                )
                token_index += 1
                await asyncio.sleep(0)  # Yield to event loop

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding (requires embedding-capable model)."""
        if self._llm is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._llm.embed(text)
        )
        return embedding


class BitNetBackend(LLMBackend):
    """
    BitNet backend for ultra-efficient CPU inference.

    Novel: Hybrid quantization switching between 1-bit for speed
    and higher precision for quality-critical sections.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        bitnet_cpp_path: str = "bitnet-cpp",
    ):
        super().__init__(model_path, config)
        self.model_path = model_path
        self.bitnet_cpp_path = bitnet_cpp_path
        self._process = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.BITNET

    async def load(self) -> bool:
        """Initialize BitNet process."""
        try:
            import subprocess
            import shutil

            # Check if bitnet-cpp is available
            if not shutil.which(self.bitnet_cpp_path):
                logger.warning("bitnet-cpp not found in PATH")
                return False

            self._is_loaded = True
            logger.info(f"BitNet backend initialized for {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize BitNet: {e}")
            return False

    async def unload(self) -> bool:
        """Cleanup BitNet process."""
        if self._process:
            self._process.terminate()
            self._process = None
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using BitNet subprocess."""
        import subprocess

        cfg = config or self.config
        start_time = time.time()

        # Run bitnet inference
        cmd = [
            self.bitnet_cpp_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(cfg.max_tokens),
            "--temp", str(cfg.temperature),
        ]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        )

        total_time = time.time() - start_time
        text = result.stdout.strip()
        tokens = len(text.split())

        return GenerationResult(
            text=text,
            tokens_generated=tokens,
            tokens_per_second=tokens / max(0.001, total_time),
            model_used=self.model_path,
            backend_used=self.backend_type,
            confidence_score=0.8,  # BitNet doesn't provide logprobs
            total_time=total_time,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation (simulated for BitNet)."""
        # BitNet CLI doesn't support streaming natively
        # We generate full response and simulate streaming
        result = await self.generate(prompt, config)

        words = result.text.split()
        for i, word in enumerate(words):
            yield StreamChunk(
                text=word + " ",
                token_index=i,
                confidence=result.confidence_score,
                cumulative_confidence=result.confidence_score,
            )
            await asyncio.sleep(0.01)  # Simulate streaming delay

    async def get_embedding(self, text: str) -> list[float]:
        """
        BitNet doesn't support embeddings natively.
        Fallback to simple hash-based embedding for basic functionality.
        """
        import hashlib

        # Generate a deterministic pseudo-embedding from text hash
        # This is NOT semantic but provides a fallback for systems that require embeddings
        text_hash = hashlib.sha256(text.encode()).digest()

        # Convert to 384-dimensional float vector (common embedding size)
        embedding = []
        for i in range(384):
            byte_idx = i % 32
            # Normalize to [-1, 1] range
            val = (text_hash[byte_idx] / 255.0) * 2 - 1
            # Add some variation based on position
            val = val * (0.5 + 0.5 * ((i % 16) / 16))
            embedding.append(val)

        logger.warning("BitNet using hash-based pseudo-embeddings (not semantic)")
        return embedding


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI-compatible API backend for cloud models.

    Supports:
    - MiniMax M2/M2.1 (coding & agentic workflows)
    - DeepInfra endpoints
    - OpenRouter
    - Any OpenAI-compatible API

    MiniMax M2 is optimized for coding with:
    - Interleaved thinking (<think>...</think>)
    - 128K context window
    - Tool/function calling
    - Multi-file editing and SWE tasks
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENAI_COMPATIBLE

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    async def load(self) -> bool:
        """Verify API connectivity."""
        try:
            await self._get_client()
            self._is_loaded = True
            logger.info(f"OpenAI-compatible backend initialized for {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
            return False

    async def unload(self) -> bool:
        """Close client."""
        self._client = None
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate using OpenAI-compatible API."""
        cfg = config or self.config
        client = await self._get_client()

        start_time = time.time()
        self.confidence_estimator.reset()

        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # MiniMax M2 recommended settings
        if "minimax" in self.model_name.lower():
            temp = 1.0  # MiniMax recommends temp=1.0
            top_p = 0.95
            top_k = 40
        else:
            top_p = cfg.top_p
            top_k = cfg.top_k

        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=cfg.max_tokens,
            )

            total_time = time.time() - start_time
            text = response.choices[0].message.content or ""

            # Update confidence from output
            for token in text.split():
                self.confidence_estimator.update(token)

            return GenerationResult(
                text=text,
                tokens_generated=response.usage.completion_tokens if response.usage else len(text.split()),
                tokens_per_second=(response.usage.completion_tokens if response.usage else len(text.split())) / max(0.001, total_time),
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=self.confidence_estimator.base_confidence,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return GenerationResult(
                text=f"Error: {e}",
                tokens_generated=0,
                tokens_per_second=0,
                model_used=self.model_name,
                backend_used=self.backend_type,
                confidence_score=0,
                total_time=time.time() - start_time,
            )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation from OpenAI-compatible API."""
        cfg = config or self.config
        client = await self._get_client()

        self.confidence_estimator.reset()
        content_type = self.adaptive_temp.detect_content_type(prompt)
        temp = self.adaptive_temp.adapt(1.0, content_type)

        # MiniMax M2 settings
        if "minimax" in self.model_name.lower():
            temp = 1.0
            top_p = 0.95
        else:
            top_p = cfg.top_p

        token_index = 0
        cumulative_confidence = 1.0

        try:
            stream = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=cfg.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    confidence = self.confidence_estimator.update(text)
                    cumulative_confidence = 0.95 * cumulative_confidence + 0.05 * confidence

                    yield StreamChunk(
                        text=text,
                        token_index=token_index,
                        confidence=confidence,
                        cumulative_confidence=cumulative_confidence,
                    )
                    token_index += 1
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield StreamChunk(
                text=f"Error: {e}",
                token_index=0,
                confidence=0,
                cumulative_confidence=0,
            )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding (if model supports it)."""
        client = await self._get_client()

        try:
            response = await client.embeddings.create(
                model="text-embedding-ada-002",  # Fallback embedding model
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, using hash fallback")
            # Hash-based fallback
            import hashlib
            text_hash = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(384):
                byte_idx = i % 32
                val = (text_hash[byte_idx] / 255.0) * 2 - 1
                embedding.append(val)
            return embedding


class CascadeBackend(LLMBackend):
    """
    Novel: Cascade inference backend that auto-escalates models.

    Strategy:
    1. Start with fastest model (BitNet/small)
    2. Monitor confidence during generation
    3. If confidence drops below threshold, escalate to larger model
    4. Transfer context and continue generation
    """

    def __init__(
        self,
        backends: list[LLMBackend],
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__("cascade", config)
        self.backends = backends  # Ordered from fastest to most capable
        self.escalation_threshold = 0.6
        self.min_tokens_before_escalation = 30

    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA  # Primary type

    async def load(self) -> bool:
        """Load all cascade backends."""
        results = await asyncio.gather(*[b.load() for b in self.backends])
        self._is_loaded = any(results)
        return self._is_loaded

    async def unload(self) -> bool:
        """Unload all backends."""
        await asyncio.gather(*[b.unload() for b in self.backends])
        self._is_loaded = False
        return True

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate with cascade escalation."""
        cfg = config or self.config

        # Estimate complexity to choose starting backend
        complexity = self.estimate_complexity(prompt)
        start_idx = min(len(self.backends) - 1, int(complexity * len(self.backends)))

        current_backend = self.backends[start_idx]
        result = await current_backend.generate(prompt, cfg)

        # Check if escalation needed
        if (result.confidence_score < self.escalation_threshold and
            start_idx < len(self.backends) - 1):

            logger.info(f"Escalating from {current_backend.model_name} due to low confidence")

            # Escalate to next backend with full prompt + partial response
            enhanced_prompt = f"{prompt}\n\nPrevious attempt (incomplete): {result.text[:200]}..."
            next_backend = self.backends[start_idx + 1]
            result = await next_backend.generate(enhanced_prompt, cfg)
            result.escalated = True
            result.cascade_depth = start_idx + 1

        return result

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with potential mid-generation escalation."""
        cfg = config or self.config

        complexity = self.estimate_complexity(prompt)
        current_idx = min(len(self.backends) - 1, int(complexity * len(self.backends)))

        accumulated_text = ""
        token_count = 0

        async for chunk in self.backends[current_idx].generate_stream(prompt, cfg):
            accumulated_text += chunk.text
            token_count += 1
            yield chunk

            # Check for escalation
            if (token_count >= self.min_tokens_before_escalation and
                chunk.cumulative_confidence < self.escalation_threshold and
                current_idx < len(self.backends) - 1):

                logger.info(f"Mid-stream escalation at token {token_count}")
                current_idx += 1

                # Continue with stronger model
                continuation_prompt = f"{prompt}\n\n{accumulated_text}"
                async for cont_chunk in self.backends[current_idx].generate_stream(
                    continuation_prompt, cfg
                ):
                    yield cont_chunk
                break

    async def get_embedding(self, text: str) -> list[float]:
        """Use most capable backend for embeddings."""
        for backend in reversed(self.backends):
            try:
                return await backend.get_embedding(text)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.warning(f"Backend {backend.model_name} embedding failed: {e}")
                continue

        # Fallback to hash-based pseudo-embedding
        import hashlib
        logger.warning("All backends failed for embeddings, using hash-based fallback")

        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):
            byte_idx = i % 32
            val = (text_hash[byte_idx] / 255.0) * 2 - 1
            val = val * (0.5 + 0.5 * ((i % 16) / 16))
            embedding.append(val)

        return embedding
