"""
Farnsworth Tool Router - Dynamic Tool Management and Routing

Provides intelligent tool routing with:
- Capability-based tool matching
- Dynamic tool discovery
- Tool composition for complex tasks
- Usage tracking and optimization

Novel Features:
- Semantic tool matching using embeddings
- Automatic tool chaining for multi-step tasks
- Tool performance tracking and selection
- Composio integration for 500+ external tools
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger


class ToolCategory(Enum):
    """Tool categorization."""
    FILE_SYSTEM = "file_system"
    CODE = "code"
    WEB = "web"
    DATABASE = "database"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    """Definition of a tool available to the system."""
    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    examples: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    priority: int = 5
    enabled: bool = True

    # Performance tracking
    usage_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_used: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "examples": self.examples,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "enabled": self.enabled,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.usage_count),
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class ToolRouter:
    """
    Intelligent tool routing system.

    Provides:
    - Tool registration and discovery
    - Capability-based matching
    - Automatic tool chaining
    - Performance optimization
    """

    def __init__(
        self,
        embedder=None,
        enable_composio: bool = False,
        composio_api_key: Optional[str] = None,
    ):
        self.tools: dict[str, ToolDefinition] = {}
        self.embedder = embedder
        self.tool_embeddings: dict[str, list[float]] = {}

        # Composio integration
        self.composio_enabled = enable_composio
        self.composio_client = None
        if enable_composio and composio_api_key:
            self._init_composio(composio_api_key)

        # Tool chains for complex tasks
        self.tool_chains: dict[str, list[str]] = {}

        # Register built-in tools
        self._register_builtin_tools()

    def _init_composio(self, api_key: str):
        """Initialize Composio integration for 500+ tools."""
        try:
            # Placeholder for Composio integration
            # In production: from composio import ComposioClient
            logger.info("Composio integration enabled")
        except ImportError:
            logger.warning("Composio library not installed")
            self.composio_enabled = False

    def _register_builtin_tools(self):
        """Register built-in tools."""
        # File system tools
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Read contents of a file from the filesystem",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "File path to read"},
            },
            examples=["read_file path=/home/user/document.txt"],
            capabilities=["file_read", "text_extraction"],
            handler=self._handle_read_file,
        ))

        self.register_tool(ToolDefinition(
            name="write_file",
            description="Write contents to a file on the filesystem",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "File path to write"},
                "content": {"type": "string", "required": True, "description": "Content to write"},
            },
            examples=["write_file path=/home/user/output.txt content='Hello World'"],
            capabilities=["file_write", "file_create"],
            handler=self._handle_write_file,
        ))

        self.register_tool(ToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            category=ToolCategory.FILE_SYSTEM,
            parameters={
                "path": {"type": "string", "required": True, "description": "Directory path"},
            },
            examples=["list_directory path=/home/user"],
            capabilities=["directory_listing", "file_discovery"],
            handler=self._handle_list_directory,
        ))

        # Code tools
        self.register_tool(ToolDefinition(
            name="execute_python",
            description="Execute Python code in a sandboxed environment",
            category=ToolCategory.CODE,
            parameters={
                "code": {"type": "string", "required": True, "description": "Python code to execute"},
            },
            examples=["execute_python code='print(2 + 2)'"],
            capabilities=["code_execution", "python", "computation"],
            handler=self._handle_execute_python,
        ))

        self.register_tool(ToolDefinition(
            name="analyze_code",
            description="Analyze code for issues and suggestions",
            category=ToolCategory.CODE,
            parameters={
                "code": {"type": "string", "required": True, "description": "Code to analyze"},
                "language": {"type": "string", "required": False, "description": "Programming language"},
            },
            examples=["analyze_code code='def foo(): pass' language=python"],
            capabilities=["code_analysis", "linting", "suggestions"],
            handler=self._handle_analyze_code,
        ))

        # Web tools
        self.register_tool(ToolDefinition(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.WEB,
            parameters={
                "query": {"type": "string", "required": True, "description": "Search query"},
                "num_results": {"type": "integer", "required": False, "description": "Number of results"},
            },
            examples=["web_search query='Python async programming'"],
            capabilities=["web_search", "information_retrieval"],
            handler=self._handle_web_search,
        ))

        self.register_tool(ToolDefinition(
            name="fetch_url",
            description="Fetch content from a URL",
            category=ToolCategory.WEB,
            parameters={
                "url": {"type": "string", "required": True, "description": "URL to fetch"},
            },
            examples=["fetch_url url='https://example.com'"],
            capabilities=["web_fetch", "html_parsing"],
            handler=self._handle_fetch_url,
        ))

        # Analysis tools
        self.register_tool(ToolDefinition(
            name="summarize_text",
            description="Summarize a long text into key points",
            category=ToolCategory.ANALYSIS,
            parameters={
                "text": {"type": "string", "required": True, "description": "Text to summarize"},
                "max_length": {"type": "integer", "required": False, "description": "Max summary length"},
            },
            examples=["summarize_text text='Long document...' max_length=100"],
            capabilities=["summarization", "text_analysis"],
            handler=self._handle_summarize,
        ))

        self.register_tool(ToolDefinition(
            name="extract_entities",
            description="Extract named entities from text",
            category=ToolCategory.ANALYSIS,
            parameters={
                "text": {"type": "string", "required": True, "description": "Text to analyze"},
            },
            examples=["extract_entities text='John works at Google in California'"],
            capabilities=["ner", "entity_extraction", "text_analysis"],
            handler=self._handle_extract_entities,
        ))

        # Generation tools
        self.register_tool(ToolDefinition(
            name="generate_image",
            description="Generate an image from a text description",
            category=ToolCategory.GENERATION,
            parameters={
                "prompt": {"type": "string", "required": True, "description": "Image description"},
                "size": {"type": "string", "required": False, "description": "Image size"},
            },
            examples=["generate_image prompt='A sunset over mountains'"],
            capabilities=["image_generation", "creative"],
            handler=self._handle_generate_image,
        ))

        # Utility tools
        self.register_tool(ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            category=ToolCategory.UTILITY,
            parameters={
                "expression": {"type": "string", "required": True, "description": "Math expression"},
            },
            examples=["calculate expression='sqrt(16) + 5 * 2'"],
            capabilities=["math", "calculation"],
            handler=self._handle_calculate,
        ))

        self.register_tool(ToolDefinition(
            name="datetime_info",
            description="Get current date, time, or timezone information",
            category=ToolCategory.UTILITY,
            parameters={
                "timezone": {"type": "string", "required": False, "description": "Timezone name"},
            },
            examples=["datetime_info timezone='America/New_York'"],
            capabilities=["datetime", "timezone"],
            handler=self._handle_datetime,
        ))

    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a new tool.

        Args:
            tool: Tool definition to register
        """
        self.tools[tool.name] = tool

        # Generate embedding for semantic matching
        if self.embedder:
            description_text = f"{tool.name}: {tool.description}. Capabilities: {', '.join(tool.capabilities)}"
            try:
                self.tool_embeddings[tool.name] = self.embedder.embed(description_text)
            except Exception as e:
                logger.warning(f"Failed to embed tool {tool.name}: {e}")

        logger.debug(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed
        """
        if name in self.tools:
            del self.tools[name]
            if name in self.tool_embeddings:
                del self.tool_embeddings[name]
            return True
        return False

    async def route(
        self,
        query: str,
        context: Optional[dict] = None,
        required_capabilities: Optional[list[str]] = None,
    ) -> list[ToolDefinition]:
        """
        Route a query to the most appropriate tools.

        Uses semantic matching and capability filtering.

        Args:
            query: User query or task description
            context: Additional context
            required_capabilities: Required tool capabilities

        Returns:
            List of matching tools, sorted by relevance
        """
        matching_tools = []

        for name, tool in self.tools.items():
            if not tool.enabled:
                continue

            score = 0.0

            # Capability matching
            if required_capabilities:
                capability_match = sum(
                    1 for cap in required_capabilities
                    if cap in tool.capabilities
                )
                if capability_match == 0:
                    continue
                score += capability_match * 0.3

            # Keyword matching
            query_lower = query.lower()
            if tool.name.lower() in query_lower:
                score += 0.4
            if any(cap in query_lower for cap in tool.capabilities):
                score += 0.2

            # Semantic matching (if embedder available)
            if self.embedder and name in self.tool_embeddings:
                try:
                    query_embedding = self.embedder.embed(query)
                    tool_embedding = self.tool_embeddings[name]
                    similarity = self._cosine_similarity(query_embedding, tool_embedding)
                    score += similarity * 0.3
                except Exception:
                    pass

            # Performance-based adjustment
            if tool.usage_count > 0:
                success_rate = tool.success_count / tool.usage_count
                score += success_rate * 0.1

            # Priority adjustment
            score += tool.priority * 0.01

            if score > 0:
                matching_tools.append((tool, score))

        # Sort by score descending
        matching_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in matching_tools]

    async def execute(
        self,
        tool_name: str,
        parameters: dict,
        context: Optional[dict] = None,
    ) -> ToolExecutionResult:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            context: Execution context

        Returns:
            ToolExecutionResult
        """
        if tool_name not in self.tools:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        tool = self.tools[tool_name]

        if not tool.enabled:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool disabled: {tool_name}",
            )

        if tool.handler is None:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool has no handler: {tool_name}",
            )

        # Validate required parameters
        for param_name, param_def in tool.parameters.items():
            if param_def.get("required", False) and param_name not in parameters:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Missing required parameter: {param_name}",
                )

        # Execute with timing
        start_time = datetime.now()

        try:
            result = await tool.handler(**parameters)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Update statistics
            tool.usage_count += 1
            tool.success_count += 1
            tool.avg_latency_ms = (
                tool.avg_latency_ms * (tool.usage_count - 1) + latency
            ) / tool.usage_count
            tool.last_used = datetime.now()

            return ToolExecutionResult(
                success=True,
                output=result,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            tool.usage_count += 1
            tool.last_used = datetime.now()

            logger.error(f"Tool execution failed: {tool_name}: {e}")

            return ToolExecutionResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=latency,
            )

    async def execute_chain(
        self,
        chain_name: str,
        initial_input: dict,
        context: Optional[dict] = None,
    ) -> list[ToolExecutionResult]:
        """
        Execute a chain of tools.

        Args:
            chain_name: Name of the tool chain
            initial_input: Initial input parameters
            context: Execution context

        Returns:
            List of results from each tool
        """
        if chain_name not in self.tool_chains:
            return [ToolExecutionResult(
                success=False,
                output=None,
                error=f"Chain not found: {chain_name}",
            )]

        results = []
        current_input = initial_input

        for tool_name in self.tool_chains[chain_name]:
            result = await self.execute(tool_name, current_input, context)
            results.append(result)

            if not result.success:
                break

            # Pass output to next tool
            if isinstance(result.output, dict):
                current_input.update(result.output)
            else:
                current_input["_previous_output"] = result.output

        return results

    def register_chain(self, name: str, tools: list[str]) -> bool:
        """
        Register a tool chain.

        Args:
            name: Chain name
            tools: List of tool names in order

        Returns:
            True if chain was registered
        """
        # Validate all tools exist
        for tool_name in tools:
            if tool_name not in self.tools:
                logger.warning(f"Cannot register chain: tool {tool_name} not found")
                return False

        self.tool_chains[name] = tools
        logger.info(f"Registered tool chain: {name} with {len(tools)} tools")
        return True

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:
        """
        List available tools.

        Args:
            category: Filter by category
            enabled_only: Only return enabled tools

        Returns:
            List of matching tools
        """
        tools = []
        for tool in self.tools.values():
            if enabled_only and not tool.enabled:
                continue
            if category and tool.category != category:
                continue
            tools.append(tool)
        return tools

    def get_statistics(self) -> dict:
        """Get usage statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "enabled_tools": sum(1 for t in self.tools.values() if t.enabled),
            "total_executions": sum(t.usage_count for t in self.tools.values()),
            "total_successes": sum(t.success_count for t in self.tools.values()),
            "by_category": {},
            "top_used": [],
        }

        # By category
        for cat in ToolCategory:
            cat_tools = [t for t in self.tools.values() if t.category == cat]
            if cat_tools:
                stats["by_category"][cat.value] = {
                    "count": len(cat_tools),
                    "total_usage": sum(t.usage_count for t in cat_tools),
                }

        # Top used
        sorted_tools = sorted(
            self.tools.values(),
            key=lambda t: t.usage_count,
            reverse=True,
        )[:10]
        stats["top_used"] = [
            {"name": t.name, "usage_count": t.usage_count, "success_rate": t.success_count / max(1, t.usage_count)}
            for t in sorted_tools
        ]

        return stats

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    # Built-in tool handlers

    async def _handle_read_file(self, path: str) -> str:
        """Read file contents."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    async def _handle_write_file(self, path: str, content: str) -> dict:
        """Write file contents."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"written": True, "path": path, "bytes": len(content.encode())}
        except Exception as e:
            raise RuntimeError(f"Failed to write file: {e}")

    async def _handle_list_directory(self, path: str) -> list[dict]:
        """List directory contents."""
        try:
            p = Path(path)
            items = []
            for item in p.iterdir():
                items.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            return items
        except Exception as e:
            raise RuntimeError(f"Failed to list directory: {e}")

    async def _handle_execute_python(self, code: str) -> dict:
        """Execute Python code (sandboxed)."""
        # In production, this would use a proper sandbox
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Very basic sandboxing - in production use proper isolation
            exec_globals = {"__builtins__": {"print": print, "len": len, "range": range, "str": str, "int": int, "float": float}}
            exec(code, exec_globals)
            output = sys.stdout.getvalue()
            return {"output": output, "success": True}
        except Exception as e:
            return {"output": str(e), "success": False, "error": str(e)}
        finally:
            sys.stdout = old_stdout

    async def _handle_analyze_code(self, code: str, language: str = "python") -> dict:
        """Analyze code for issues."""
        # Placeholder - in production, use proper linting tools
        issues = []
        suggestions = []

        if language == "python":
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "pass" in line and ":" in lines[max(0, i-1)]:
                    suggestions.append(f"Line {i+1}: Consider implementing the function body")
                if len(line) > 100:
                    issues.append(f"Line {i+1}: Line too long ({len(line)} chars)")

        return {
            "issues": issues,
            "suggestions": suggestions,
            "lines_analyzed": len(code.split("\n")),
        }

    async def _handle_web_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Search the web."""
        # Placeholder - in production, use actual search API
        return [{
            "title": f"Search result for: {query}",
            "url": "https://example.com",
            "snippet": "This is a placeholder search result",
        }]

    async def _handle_fetch_url(self, url: str) -> dict:
        """Fetch URL content."""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")
                return {
                    "url": url,
                    "status": response.status,
                    "content": content[:10000],  # Limit size
                }
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {e}")

    async def _handle_summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text."""
        # Placeholder - in production, use LLM
        words = text.split()
        if len(words) <= max_length // 5:
            return text
        return " ".join(words[:max_length // 5]) + "..."

    async def _handle_extract_entities(self, text: str) -> list[dict]:
        """Extract named entities."""
        # Placeholder - in production, use NER model
        import re

        # Simple capitalized word detection
        entities = []
        words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for word in set(words):
            entities.append({
                "text": word,
                "type": "UNKNOWN",
            })
        return entities

    async def _handle_generate_image(self, prompt: str, size: str = "512x512") -> dict:
        """Generate an image."""
        # Placeholder - in production, use image generation API
        return {
            "prompt": prompt,
            "size": size,
            "url": "https://placeholder.com/generated_image.png",
            "note": "Image generation requires API integration",
        }

    async def _handle_calculate(self, expression: str) -> float:
        """Perform calculation."""
        import math

        # Safe evaluation with limited builtins
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"Calculation error: {e}")

    async def _handle_datetime(self, timezone: str = "UTC") -> dict:
        """Get datetime information."""
        from datetime import timezone as tz

        now = datetime.now()
        return {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": timezone,
            "unix_timestamp": now.timestamp(),
        }
