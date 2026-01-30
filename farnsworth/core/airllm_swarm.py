"""
AirLLM Side Swarm - Background Processing During Idle Time

Uses AirLLM to run large models (70B-405B) for:
- Code review and analysis
- Document summarization
- Deep research tasks
- Background reasoning

Runs during idle periods to avoid interfering with main swarm.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from loguru import logger
import threading


class AirLLMSideSwarm:
    """
    Background swarm using AirLLM for heavy processing during idle time.

    Features:
    - Idle detection (only runs when main swarm is quiet)
    - Task queue with priorities
    - Results caching
    - Non-blocking execution
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        idle_threshold_seconds: float = 30.0,
        max_queue_size: int = 50
    ):
        self.model_name = model_name
        self.idle_threshold = idle_threshold_seconds
        self.max_queue_size = max_queue_size

        # State
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.is_processing = False
        self.last_main_activity = datetime.now()

        # Task queue: (priority, timestamp, task_id, task_data)
        self.task_queue: deque = deque(maxlen=max_queue_size)
        self.results: Dict[str, Any] = {}
        self.task_counter = 0

        # Background worker
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self.on_result_ready: Optional[Callable] = None

        logger.info(f"AirLLMSideSwarm initialized with model: {model_name}")

    async def load_model(self):
        """Load the AirLLM model (lazy loading on first use)."""
        if self.is_loaded:
            return True

        try:
            logger.info(f"Loading AirLLM model: {self.model_name}...")

            # Import here to avoid loading at startup
            from airllm import AutoModel

            # Use compression for better performance
            self.model = AutoModel.from_pretrained(
                self.model_name,
                compression="4bit"  # 4-bit quantization for speed
            )

            self.is_loaded = True
            logger.success(f"AirLLM model loaded: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load AirLLM model: {e}")
            return False

    def record_main_activity(self):
        """Record activity from main swarm (resets idle timer)."""
        self.last_main_activity = datetime.now()

    def is_idle(self) -> bool:
        """Check if main swarm has been idle long enough."""
        idle_time = (datetime.now() - self.last_main_activity).total_seconds()
        return idle_time >= self.idle_threshold

    def queue_task(
        self,
        task_type: str,
        prompt: str,
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Queue a task for background processing.

        Args:
            task_type: Type of task (code_review, summarize, research, etc.)
            prompt: The prompt to process
            priority: 1-10 (1 = highest priority)
            metadata: Optional metadata to include with result

        Returns:
            task_id: ID to retrieve results later
        """
        self.task_counter += 1
        task_id = f"air_{self.task_counter}_{int(time.time())}"

        task = {
            "id": task_id,
            "type": task_type,
            "prompt": prompt,
            "priority": priority,
            "metadata": metadata or {},
            "queued_at": datetime.now().isoformat(),
            "status": "queued"
        }

        self.task_queue.append((priority, time.time(), task_id, task))
        self.results[task_id] = {"status": "queued", "task": task}

        logger.info(f"AirLLM task queued: {task_id} ({task_type})")
        return task_id

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get result for a task (or status if still processing)."""
        return self.results.get(task_id)

    async def _process_task(self, task: Dict) -> Dict:
        """Process a single task."""
        task_id = task["id"]
        self.results[task_id]["status"] = "processing"
        self.is_processing = True

        try:
            logger.info(f"AirLLM processing: {task_id}")
            start_time = time.time()

            # Build prompt based on task type
            system_prompts = {
                "code_review": "You are an expert code reviewer. Analyze the code for bugs, security issues, and improvements.",
                "summarize": "You are a summarization expert. Provide a clear, concise summary.",
                "research": "You are a research assistant. Provide comprehensive analysis with sources.",
                "analyze": "You are an analytical expert. Break down the problem and provide insights.",
                "default": "You are a helpful AI assistant."
            }

            system = system_prompts.get(task["type"], system_prompts["default"])
            full_prompt = f"{system}\n\nUser: {task['prompt']}\n\nAssistant:"

            # Generate with AirLLM
            if not self.is_loaded:
                await self.load_model()

            if self.model:
                # Run in thread to not block
                def generate():
                    input_ids = self.model.tokenizer(
                        full_prompt,
                        return_tensors="pt"
                    ).input_ids

                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=1024,
                        temperature=0.7,
                        do_sample=True
                    )

                    return self.model.tokenizer.decode(
                        output[0][len(input_ids[0]):],
                        skip_special_tokens=True
                    )

                # Run generation in thread
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, generate)
            else:
                response = "[AirLLM model not loaded - using placeholder]"

            elapsed = time.time() - start_time

            result = {
                "status": "completed",
                "task": task,
                "response": response,
                "elapsed_seconds": elapsed,
                "completed_at": datetime.now().isoformat()
            }

            self.results[task_id] = result
            logger.success(f"AirLLM completed: {task_id} in {elapsed:.1f}s")

            # Callback if set
            if self.on_result_ready:
                try:
                    await self.on_result_ready(result)
                except Exception as e:
                    logger.error(f"Result callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"AirLLM task error: {e}")
            self.results[task_id] = {
                "status": "error",
                "task": task,
                "error": str(e)
            }
            return self.results[task_id]
        finally:
            self.is_processing = False

    async def _worker_loop(self):
        """Background worker that processes tasks during idle time."""
        logger.info("AirLLM worker started")

        while self._running:
            try:
                # Wait for idle
                while not self.is_idle() and self._running:
                    await asyncio.sleep(5)

                if not self._running:
                    break

                # Check for tasks
                if not self.task_queue:
                    await asyncio.sleep(10)
                    continue

                # Sort by priority and get highest priority task
                sorted_tasks = sorted(self.task_queue, key=lambda x: (x[0], x[1]))
                if sorted_tasks:
                    priority, timestamp, task_id, task = sorted_tasks[0]
                    self.task_queue.remove((priority, timestamp, task_id, task))

                    # Double-check still idle before processing
                    if self.is_idle():
                        await self._process_task(task)
                    else:
                        # Re-queue if no longer idle
                        self.task_queue.appendleft((priority, timestamp, task_id, task))
                        logger.debug("AirLLM paused - main swarm active")

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AirLLM worker error: {e}")
                await asyncio.sleep(5)

        logger.info("AirLLM worker stopped")

    async def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("AirLLM side swarm started")

    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("AirLLM side swarm stopped")

    def get_stats(self) -> Dict:
        """Get swarm statistics."""
        queued = len([r for r in self.results.values() if r.get("status") == "queued"])
        completed = len([r for r in self.results.values() if r.get("status") == "completed"])
        errors = len([r for r in self.results.values() if r.get("status") == "error"])

        return {
            "model": self.model_name,
            "is_loaded": self.is_loaded,
            "is_processing": self.is_processing,
            "is_idle": self.is_idle(),
            "idle_threshold_seconds": self.idle_threshold,
            "queue_size": len(self.task_queue),
            "tasks": {
                "queued": queued,
                "completed": completed,
                "errors": errors,
                "total": len(self.results)
            },
            "running": self._running
        }


# Global instance
_airllm_swarm: Optional[AirLLMSideSwarm] = None


def get_airllm_swarm() -> Optional[AirLLMSideSwarm]:
    """Get the global AirLLM swarm instance."""
    return _airllm_swarm


async def initialize_airllm_swarm(
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
    idle_threshold: float = 30.0
) -> AirLLMSideSwarm:
    """Initialize and start the AirLLM side swarm."""
    global _airllm_swarm

    if _airllm_swarm is None:
        _airllm_swarm = AirLLMSideSwarm(
            model_name=model_name,
            idle_threshold_seconds=idle_threshold
        )
        await _airllm_swarm.start()
        logger.success("AirLLM side swarm initialized and started")

    return _airllm_swarm


async def queue_background_task(
    task_type: str,
    prompt: str,
    priority: int = 5
) -> Optional[str]:
    """Queue a task for background processing."""
    if _airllm_swarm:
        return _airllm_swarm.queue_task(task_type, prompt, priority)
    return None
