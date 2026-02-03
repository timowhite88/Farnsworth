"""
Farnsworth n8n Integration - Full Workflow Automation.

"I can wire anything into anything! I'm the Professor!"

Complete n8n API integration:
- Workflow management (CRUD)
- Workflow execution and monitoring
- Webhook management
- Execution history and logs
- Credentials management
- Tags and organization
- Node management
- Error handling and retry
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from farnsworth.core.nexus import nexus, Signal, SignalType


class WorkflowStatus(Enum):
    """n8n workflow status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class ExecutionStatus(Enum):
    """n8n execution status."""
    NEW = "new"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WAITING = "waiting"
    CANCELED = "canceled"


@dataclass
class N8nWorkflow:
    """Structured n8n workflow data."""
    id: str
    name: str
    active: bool = False
    tags: List[str] = field(default_factory=list)
    nodes: List[Dict] = field(default_factory=list)
    connections: Dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    settings: Dict = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def trigger_nodes(self) -> List[Dict]:
        """Get trigger nodes (entry points)."""
        return [n for n in self.nodes if n.get("type", "").endswith("Trigger")]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "active": self.active,
            "tags": self.tags,
            "node_count": self.node_count,
            "trigger_types": [n.get("type", "") for n in self.trigger_nodes],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class N8nExecution:
    """Structured n8n execution data."""
    id: str
    workflow_id: str
    workflow_name: str = ""
    status: ExecutionStatus = ExecutionStatus.NEW
    mode: str = "manual"  # manual, webhook, trigger, etc.
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    data: Dict = field(default_factory=dict)
    error: Optional[str] = None
    retry_of: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "mode": self.mode,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error
        }


@dataclass
class N8nWebhook:
    """Structured webhook data."""
    id: str
    workflow_id: str
    path: str
    method: str = "GET"
    node_name: str = ""
    is_active: bool = True

    @property
    def full_path(self) -> str:
        return f"/webhook/{self.path}"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "path": self.path,
            "full_path": self.full_path,
            "method": self.method,
            "node_name": self.node_name,
            "is_active": self.is_active
        }


class N8nProvider(ExternalProvider):
    """
    Full-featured n8n workflow automation integration.

    Features:
    - Complete workflow CRUD
    - Execution management and monitoring
    - Webhook creation and management
    - Execution history with filtering
    - Tag-based organization
    - Credential management
    - Error handling and retry
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(IntegrationConfig(name="n8n", api_key=api_key))
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 30  # seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "X-N8N-API-KEY": self.config.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        use_cache: bool = False
    ) -> Optional[Dict]:
        """Make an API request."""
        url = f"{self.base_url}/api/v1{endpoint}"
        cache_key = f"{method}:{endpoint}:{str(params)}"

        if use_cache and method == "GET" and cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return cached_data

        try:
            session = await self._get_session()
            async with session.request(method, url, json=data, params=params) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if use_cache and method == "GET":
                        self._cache[cache_key] = (result, datetime.now().timestamp())
                    return result
                elif resp.status == 401:
                    logger.error("n8n: Authentication failed")
                    self.status = ConnectionStatus.ERROR
                    return None
                elif resp.status == 404:
                    logger.debug(f"n8n: Resource not found: {endpoint}")
                    return None
                else:
                    error_text = await resp.text()
                    logger.error(f"n8n API error {resp.status}: {error_text[:200]}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"n8n connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"n8n request error: {e}")
            return None

    async def connect(self) -> bool:
        """Connect to n8n API."""
        if not self.config.api_key or not self.base_url:
            logger.warning("n8n: API key or base URL not provided")
            return False

        try:
            # Test connection by listing workflows
            result = await self._request("GET", "/workflows", params={"limit": 1})
            if result is not None:
                logger.info("n8n: Connected successfully")
                self.status = ConnectionStatus.CONNECTED
                return True
            else:
                logger.error("n8n: Connection test failed")
                self.status = ConnectionStatus.ERROR
                return False
        except Exception as e:
            logger.error(f"n8n connection error: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self):
        """Disconnect and cleanup."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._cache.clear()
        self.status = ConnectionStatus.DISCONNECTED

    async def sync(self):
        """Sync workflow data."""
        if self.status != ConnectionStatus.CONNECTED:
            return

        try:
            workflows = await self.list_workflows()
            self._cache["workflows"] = (workflows, datetime.now().timestamp())
            logger.debug(f"n8n: Synced {len(workflows)} workflows")
        except Exception as e:
            logger.warning(f"n8n sync error: {e}")

    # ==================== WORKFLOWS ====================

    async def list_workflows(
        self,
        active: bool = None,
        tags: List[str] = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[N8nWorkflow]:
        """
        List all workflows.

        Args:
            active: Filter by active status
            tags: Filter by tags
            limit: Maximum results
            cursor: Pagination cursor
        """
        params = {"limit": min(limit, 250)}
        if active is not None:
            params["active"] = str(active).lower()
        if tags:
            params["tags"] = ",".join(tags)
        if cursor:
            params["cursor"] = cursor

        result = await self._request("GET", "/workflows", params=params)
        if not result:
            return []

        workflows = []
        for wf in result.get("data", []):
            workflows.append(self._parse_workflow(wf))

        return workflows

    async def get_workflow(self, workflow_id: str) -> Optional[N8nWorkflow]:
        """Get a specific workflow by ID."""
        result = await self._request("GET", f"/workflows/{workflow_id}")
        if result:
            return self._parse_workflow(result)
        return None

    async def create_workflow(
        self,
        name: str,
        nodes: List[Dict] = None,
        connections: Dict = None,
        active: bool = False,
        settings: Dict = None,
        tags: List[str] = None
    ) -> Optional[N8nWorkflow]:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            nodes: List of node configurations
            connections: Node connections
            active: Whether to activate immediately
            settings: Workflow settings
            tags: Tags to apply
        """
        data = {
            "name": name,
            "nodes": nodes or [],
            "connections": connections or {},
            "active": active,
            "settings": settings or {}
        }

        result = await self._request("POST", "/workflows", data=data)
        if result:
            workflow = self._parse_workflow(result)

            # Apply tags if provided
            if tags and workflow:
                await self.add_workflow_tags(workflow.id, tags)

            return workflow
        return None

    async def update_workflow(
        self,
        workflow_id: str,
        name: str = None,
        nodes: List[Dict] = None,
        connections: Dict = None,
        active: bool = None,
        settings: Dict = None
    ) -> Optional[N8nWorkflow]:
        """Update an existing workflow."""
        data = {}
        if name is not None:
            data["name"] = name
        if nodes is not None:
            data["nodes"] = nodes
        if connections is not None:
            data["connections"] = connections
        if active is not None:
            data["active"] = active
        if settings is not None:
            data["settings"] = settings

        result = await self._request("PATCH", f"/workflows/{workflow_id}", data=data)
        if result:
            return self._parse_workflow(result)
        return None

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        result = await self._request("DELETE", f"/workflows/{workflow_id}")
        return result is not None

    async def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow."""
        result = await self._request("POST", f"/workflows/{workflow_id}/activate")
        return result is not None

    async def deactivate_workflow(self, workflow_id: str) -> bool:
        """Deactivate a workflow."""
        result = await self._request("POST", f"/workflows/{workflow_id}/deactivate")
        return result is not None

    # ==================== EXECUTIONS ====================

    async def execute_workflow(
        self,
        workflow_id: str,
        data: Dict = None,
        wait: bool = False
    ) -> Optional[N8nExecution]:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow to execute
            data: Input data for the workflow
            wait: Whether to wait for completion
        """
        body = {}
        if data:
            body["data"] = data

        result = await self._request("POST", f"/workflows/{workflow_id}/run", data=body)
        if result:
            return self._parse_execution(result)
        return None

    async def get_execution(self, execution_id: str) -> Optional[N8nExecution]:
        """Get execution details."""
        result = await self._request("GET", f"/executions/{execution_id}")
        if result:
            return self._parse_execution(result)
        return None

    async def list_executions(
        self,
        workflow_id: str = None,
        status: ExecutionStatus = None,
        limit: int = 100,
        include_data: bool = False
    ) -> List[N8nExecution]:
        """
        List executions.

        Args:
            workflow_id: Filter by workflow
            status: Filter by status
            limit: Maximum results
            include_data: Include execution data
        """
        params = {"limit": min(limit, 250)}
        if workflow_id:
            params["workflowId"] = workflow_id
        if status:
            params["status"] = status.value
        if include_data:
            params["includeData"] = "true"

        result = await self._request("GET", "/executions", params=params)
        if not result:
            return []

        return [self._parse_execution(ex) for ex in result.get("data", [])]

    async def delete_execution(self, execution_id: str) -> bool:
        """Delete an execution."""
        result = await self._request("DELETE", f"/executions/{execution_id}")
        return result is not None

    async def retry_execution(self, execution_id: str) -> Optional[N8nExecution]:
        """Retry a failed execution."""
        result = await self._request("POST", f"/executions/{execution_id}/retry")
        if result:
            return self._parse_execution(result)
        return None

    async def stop_execution(self, execution_id: str) -> bool:
        """Stop a running execution."""
        result = await self._request("POST", f"/executions/{execution_id}/stop")
        return result is not None

    async def get_execution_stats(self, days: int = 7) -> Dict:
        """Get execution statistics."""
        executions = await self.list_executions(limit=250)

        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in executions if e.started_at and e.started_at > cutoff]

        success = len([e for e in recent if e.status == ExecutionStatus.SUCCESS])
        error = len([e for e in recent if e.status == ExecutionStatus.ERROR])
        total = len(recent)

        return {
            "total": total,
            "success": success,
            "error": error,
            "success_rate": (success / total * 100) if total > 0 else 0,
            "period_days": days
        }

    # ==================== WEBHOOKS ====================

    async def get_webhooks(self, workflow_id: str = None) -> List[N8nWebhook]:
        """Get registered webhooks."""
        # n8n stores webhooks in workflow nodes
        workflows = await self.list_workflows(active=True)
        webhooks = []

        for wf in workflows:
            if workflow_id and wf.id != workflow_id:
                continue

            for node in wf.nodes:
                if "Webhook" in node.get("type", ""):
                    params = node.get("parameters", {})
                    webhooks.append(N8nWebhook(
                        id=f"{wf.id}_{node.get('name', '')}",
                        workflow_id=wf.id,
                        path=params.get("path", ""),
                        method=params.get("httpMethod", "GET"),
                        node_name=node.get("name", ""),
                        is_active=wf.active
                    ))

        return webhooks

    async def trigger_webhook(
        self,
        path: str,
        method: str = "POST",
        data: Dict = None,
        headers: Dict = None
    ) -> Optional[Dict]:
        """
        Trigger a webhook.

        Args:
            path: Webhook path (without /webhook/ prefix)
            method: HTTP method
            data: Request body
            headers: Additional headers
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}/webhook/{path.lstrip('/')}"

            request_headers = dict(headers or {})

            async with session.request(method, url, json=data, headers=request_headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"Webhook trigger failed: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Webhook trigger error: {e}")
            return None

    # ==================== TAGS ====================

    async def list_tags(self) -> List[Dict]:
        """List all tags."""
        result = await self._request("GET", "/tags")
        if result:
            return result.get("data", [])
        return []

    async def create_tag(self, name: str) -> Optional[Dict]:
        """Create a new tag."""
        result = await self._request("POST", "/tags", data={"name": name})
        return result

    async def add_workflow_tags(self, workflow_id: str, tag_ids: List[str]) -> bool:
        """Add tags to a workflow."""
        result = await self._request(
            "PUT",
            f"/workflows/{workflow_id}/tags",
            data={"tagIds": tag_ids}
        )
        return result is not None

    # ==================== CREDENTIALS ====================

    async def list_credentials(self, type_name: str = None) -> List[Dict]:
        """List credentials (metadata only, not secrets)."""
        params = {}
        if type_name:
            params["type"] = type_name

        result = await self._request("GET", "/credentials", params=params)
        if result:
            return result.get("data", [])
        return []

    async def create_credential(
        self,
        name: str,
        type_name: str,
        data: Dict
    ) -> Optional[Dict]:
        """Create a new credential."""
        result = await self._request("POST", "/credentials", data={
            "name": name,
            "type": type_name,
            "data": data
        })
        return result

    async def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        result = await self._request("DELETE", f"/credentials/{credential_id}")
        return result is not None

    # ==================== NODE BUILDERS ====================

    @staticmethod
    def build_webhook_trigger(
        path: str,
        method: str = "POST",
        name: str = "Webhook"
    ) -> Dict:
        """Build a webhook trigger node."""
        return {
            "name": name,
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [250, 300],
            "parameters": {
                "path": path,
                "httpMethod": method,
                "responseMode": "onReceived",
                "responseData": "allEntries"
            }
        }

    @staticmethod
    def build_http_request(
        url: str,
        method: str = "GET",
        name: str = "HTTP Request",
        body: Dict = None,
        headers: Dict = None
    ) -> Dict:
        """Build an HTTP request node."""
        node = {
            "name": name,
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [450, 300],
            "parameters": {
                "method": method,
                "url": url,
                "options": {}
            }
        }

        if body:
            node["parameters"]["body"] = body
        if headers:
            node["parameters"]["headerParameters"] = {
                "parameters": [{"name": k, "value": v} for k, v in headers.items()]
            }

        return node

    @staticmethod
    def build_code_node(
        code: str,
        name: str = "Code",
        language: str = "javaScript"
    ) -> Dict:
        """Build a code execution node."""
        return {
            "name": name,
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [650, 300],
            "parameters": {
                "jsCode": code,
                "mode": "runOnceForAllItems"
            }
        }

    @staticmethod
    def build_if_node(
        condition: str,
        name: str = "IF"
    ) -> Dict:
        """Build an IF conditional node."""
        return {
            "name": name,
            "type": "n8n-nodes-base.if",
            "typeVersion": 1,
            "position": [650, 300],
            "parameters": {
                "conditions": {
                    "string": [{
                        "value1": condition,
                        "operation": "isNotEmpty"
                    }]
                }
            }
        }

    @staticmethod
    def build_schedule_trigger(
        cron: str = "0 9 * * *",
        name: str = "Schedule"
    ) -> Dict:
        """Build a scheduled trigger node."""
        return {
            "name": name,
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1,
            "position": [250, 300],
            "parameters": {
                "rule": {
                    "interval": [{
                        "triggerAtHour": 9,
                        "triggerAtMinute": 0
                    }]
                }
            }
        }

    # ==================== UTILITIES ====================

    def _parse_workflow(self, data: Dict) -> N8nWorkflow:
        """Parse workflow response into N8nWorkflow."""
        created_at = None
        updated_at = None

        if data.get("createdAt"):
            try:
                created_at = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
            except:
                pass

        if data.get("updatedAt"):
            try:
                updated_at = datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00"))
            except:
                pass

        return N8nWorkflow(
            id=data.get("id", ""),
            name=data.get("name", ""),
            active=data.get("active", False),
            tags=[t.get("name", "") for t in data.get("tags", [])],
            nodes=data.get("nodes", []),
            connections=data.get("connections", {}),
            created_at=created_at,
            updated_at=updated_at,
            settings=data.get("settings", {})
        )

    def _parse_execution(self, data: Dict) -> N8nExecution:
        """Parse execution response into N8nExecution."""
        started_at = None
        finished_at = None

        if data.get("startedAt"):
            try:
                started_at = datetime.fromisoformat(data["startedAt"].replace("Z", "+00:00"))
            except:
                pass

        if data.get("stoppedAt"):
            try:
                finished_at = datetime.fromisoformat(data["stoppedAt"].replace("Z", "+00:00"))
            except:
                pass

        status_map = {
            "new": ExecutionStatus.NEW,
            "running": ExecutionStatus.RUNNING,
            "success": ExecutionStatus.SUCCESS,
            "error": ExecutionStatus.ERROR,
            "waiting": ExecutionStatus.WAITING,
            "canceled": ExecutionStatus.CANCELED
        }

        return N8nExecution(
            id=data.get("id", ""),
            workflow_id=data.get("workflowId", ""),
            workflow_name=data.get("workflowName", ""),
            status=status_map.get(data.get("status", ""), ExecutionStatus.NEW),
            mode=data.get("mode", "manual"),
            started_at=started_at,
            finished_at=finished_at,
            data=data.get("data", {}),
            error=data.get("error", {}).get("message") if data.get("error") else None,
            retry_of=data.get("retryOf")
        )

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute an action (legacy interface)."""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("n8n not connected")

        action_map = {
            "list_workflows": lambda p: self.list_workflows(),
            "get_workflow": lambda p: self.get_workflow(p.get("workflow_id")),
            "trigger_workflow": lambda p: self.execute_workflow(
                p.get("workflow_id"), p.get("data")
            ),
            "activate_workflow": lambda p: self.activate_workflow(p.get("workflow_id")),
            "deactivate_workflow": lambda p: self.deactivate_workflow(p.get("workflow_id")),
            "list_executions": lambda p: self.list_executions(p.get("workflow_id")),
            "get_execution": lambda p: self.get_execution(p.get("execution_id")),
            "trigger_webhook": lambda p: self.trigger_webhook(
                p.get("path"), p.get("method", "POST"), p.get("data")
            ),
        }

        if action in action_map:
            return await action_map[action](params)
        else:
            raise ValueError(f"Unknown action: {action}")


# ==================== SKILL INTERFACE ====================

class N8nSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        import os
        self.provider = N8nProvider(
            api_key=api_key or os.environ.get("N8N_API_KEY", ""),
            base_url=base_url or os.environ.get("N8N_URL", "http://localhost:5678")
        )
        self._connected = False

    async def connect(self) -> bool:
        """Connect to n8n."""
        self._connected = await self.provider.connect()
        return self._connected

    async def list_workflows(self) -> List[Dict]:
        """List all workflows."""
        if not self._connected:
            await self.connect()
        workflows = await self.provider.list_workflows()
        return [wf.to_dict() for wf in workflows]

    async def run_workflow(self, workflow_id: str, data: Dict = None) -> Dict:
        """Execute a workflow."""
        if not self._connected:
            await self.connect()
        execution = await self.provider.execute_workflow(workflow_id, data)
        return execution.to_dict() if execution else {}

    async def get_status(self, workflow_id: str) -> Dict:
        """Get workflow status."""
        if not self._connected:
            await self.connect()
        workflow = await self.provider.get_workflow(workflow_id)
        return workflow.to_dict() if workflow else {}

    async def trigger_webhook(self, path: str, data: Dict = None) -> Dict:
        """Trigger a webhook."""
        if not self._connected:
            await self.connect()
        result = await self.provider.trigger_webhook(path, "POST", data)
        return result or {}

    async def get_executions(self, workflow_id: str = None, limit: int = 10) -> List[Dict]:
        """Get recent executions."""
        if not self._connected:
            await self.connect()
        executions = await self.provider.list_executions(workflow_id, limit=limit)
        return [ex.to_dict() for ex in executions]

    async def get_stats(self) -> Dict:
        """Get execution statistics."""
        if not self._connected:
            await self.connect()
        return await self.provider.get_execution_stats()


# Global instance (lazy initialization)
n8n_skill: Optional[N8nSkill] = None


def get_n8n_skill(api_key: str = None, base_url: str = None) -> N8nSkill:
    """Get or create the n8n skill instance."""
    global n8n_skill
    if n8n_skill is None:
        n8n_skill = N8nSkill(api_key, base_url)
    return n8n_skill
