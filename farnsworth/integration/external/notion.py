"""
Farnsworth Notion Integration - Full-Featured Implementation.

"I keep my diagrams in a digital brain now. A REALLY organized one."

Complete Notion API integration:
- Database CRUD operations
- Page management with blocks
- Property queries and filtering
- Block content manipulation
- Comments and discussions
- Search across workspace
- Template duplication
- User and team management
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum

try:
    from notion_client import AsyncClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    logger.warning("notion-client not installed. Run: pip install notion-client")

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from farnsworth.core.nexus import nexus, Signal, SignalType


class PropertyType(Enum):
    """Notion property types."""
    TITLE = "title"
    RICH_TEXT = "rich_text"
    NUMBER = "number"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    CHECKBOX = "checkbox"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone_number"
    FORMULA = "formula"
    RELATION = "relation"
    ROLLUP = "rollup"
    PEOPLE = "people"
    FILES = "files"
    CREATED_TIME = "created_time"
    LAST_EDITED_TIME = "last_edited_time"
    STATUS = "status"


class BlockType(Enum):
    """Notion block types."""
    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    BULLETED_LIST = "bulleted_list_item"
    NUMBERED_LIST = "numbered_list_item"
    TODO = "to_do"
    TOGGLE = "toggle"
    CODE = "code"
    QUOTE = "quote"
    CALLOUT = "callout"
    DIVIDER = "divider"
    TABLE = "table"
    IMAGE = "image"
    BOOKMARK = "bookmark"
    EMBED = "embed"


@dataclass
class NotionPage:
    """Structured Notion page data."""
    id: str
    title: str
    url: str
    parent_id: str = ""
    parent_type: str = ""  # database, page, workspace
    icon: Optional[str] = None
    cover: Optional[str] = None
    properties: Dict = field(default_factory=dict)
    created_time: Optional[datetime] = None
    last_edited_time: Optional[datetime] = None
    archived: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "parent_id": self.parent_id,
            "parent_type": self.parent_type,
            "icon": self.icon,
            "cover": self.cover,
            "properties": self.properties,
            "archived": self.archived
        }


@dataclass
class NotionDatabase:
    """Structured Notion database data."""
    id: str
    title: str
    url: str
    description: str = ""
    properties: Dict = field(default_factory=dict)
    is_inline: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "properties": list(self.properties.keys()),
            "is_inline": self.is_inline
        }


class NotionProvider(ExternalProvider):
    """
    Full-featured Notion API integration.

    Features:
    - Database operations (query, create, update)
    - Page management (create, update, archive)
    - Block manipulation (add, update, delete, reorder)
    - Search across workspace
    - Comment management
    - User listing
    """

    def __init__(self, token: str):
        super().__init__(IntegrationConfig(name="notion", api_key=token))
        self.client: Optional[AsyncClient] = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds

    async def connect(self) -> bool:
        """Connect to Notion API."""
        if not NOTION_AVAILABLE:
            logger.error("Notion: notion-client package not installed")
            return False

        if not self.config.api_key:
            logger.warning("Notion: No API token provided")
            return False

        try:
            self.client = AsyncClient(auth=self.config.api_key)
            # Test connection by listing users
            await self.client.users.list()

            logger.info("Notion: Connected successfully")
            self.status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Notion connection failed: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self):
        """Disconnect from Notion."""
        self.client = None
        self.status = ConnectionStatus.DISCONNECTED
        self._cache.clear()

    async def sync(self):
        """Sync workspace data."""
        if self.status != ConnectionStatus.CONNECTED:
            return

        try:
            # Refresh cached databases list
            databases = await self.list_databases()
            self._cache["databases"] = (databases, datetime.now().timestamp())
            logger.debug(f"Notion: Synced {len(databases)} databases")
        except Exception as e:
            logger.warning(f"Notion sync error: {e}")

    # ==================== SEARCH ====================

    async def search(
        self,
        query: str,
        filter_type: str = None,  # "page" or "database"
        sort_direction: str = "descending"
    ) -> List[Dict]:
        """
        Search across the entire workspace.

        Args:
            query: Search query string
            filter_type: Filter by "page" or "database"
            sort_direction: "ascending" or "descending" by last edited
        """
        if not self.client:
            return []

        try:
            params = {"query": query}

            if filter_type:
                params["filter"] = {"property": "object", "value": filter_type}

            params["sort"] = {
                "direction": sort_direction,
                "timestamp": "last_edited_time"
            }

            results = await self.client.search(**params)
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Notion search error: {e}")
            return []

    # ==================== DATABASES ====================

    async def list_databases(self) -> List[NotionDatabase]:
        """List all accessible databases."""
        if not self.client:
            return []

        try:
            results = await self.client.search(filter={"property": "object", "value": "database"})
            databases = []

            for db in results.get("results", []):
                title = ""
                if db.get("title"):
                    title = "".join(t.get("plain_text", "") for t in db["title"])

                databases.append(NotionDatabase(
                    id=db["id"],
                    title=title,
                    url=db.get("url", ""),
                    description=db.get("description", [{}])[0].get("plain_text", "") if db.get("description") else "",
                    properties=db.get("properties", {}),
                    is_inline=db.get("is_inline", False)
                ))

            return databases
        except Exception as e:
            logger.error(f"Notion list databases error: {e}")
            return []

    async def get_database(self, database_id: str) -> Optional[NotionDatabase]:
        """Get database details by ID."""
        if not self.client:
            return None

        try:
            db = await self.client.databases.retrieve(database_id)
            title = "".join(t.get("plain_text", "") for t in db.get("title", []))

            return NotionDatabase(
                id=db["id"],
                title=title,
                url=db.get("url", ""),
                description=db.get("description", [{}])[0].get("plain_text", "") if db.get("description") else "",
                properties=db.get("properties", {}),
                is_inline=db.get("is_inline", False)
            )
        except Exception as e:
            logger.error(f"Notion get database error: {e}")
            return None

    async def query_database(
        self,
        database_id: str,
        filter: Dict = None,
        sorts: List[Dict] = None,
        page_size: int = 100
    ) -> List[NotionPage]:
        """
        Query a database with optional filters and sorting.

        Args:
            database_id: The database to query
            filter: Notion filter object
            sorts: List of sort objects
            page_size: Number of results per page (max 100)
        """
        if not self.client:
            return []

        try:
            params = {"database_id": database_id, "page_size": min(page_size, 100)}
            if filter:
                params["filter"] = filter
            if sorts:
                params["sorts"] = sorts

            results = await self.client.databases.query(**params)
            pages = []

            for page in results.get("results", []):
                pages.append(self._parse_page(page))

            return pages
        except Exception as e:
            logger.error(f"Notion query database error: {e}")
            return []

    async def create_database(
        self,
        parent_page_id: str,
        title: str,
        properties: Dict[str, PropertyType],
        is_inline: bool = False
    ) -> Optional[NotionDatabase]:
        """
        Create a new database.

        Args:
            parent_page_id: Parent page ID
            title: Database title
            properties: Dict of property names to PropertyType enums
            is_inline: Whether database is inline
        """
        if not self.client:
            return None

        try:
            # Build properties schema
            props = {"Name": {"title": {}}}  # Required title property
            for name, prop_type in properties.items():
                if name != "Name":
                    props[name] = {prop_type.value: {}}

            db = await self.client.databases.create(
                parent={"type": "page_id", "page_id": parent_page_id},
                title=[{"type": "text", "text": {"content": title}}],
                properties=props,
                is_inline=is_inline
            )

            return NotionDatabase(
                id=db["id"],
                title=title,
                url=db.get("url", ""),
                properties=db.get("properties", {}),
                is_inline=is_inline
            )
        except Exception as e:
            logger.error(f"Notion create database error: {e}")
            return None

    # ==================== PAGES ====================

    async def get_page(self, page_id: str) -> Optional[NotionPage]:
        """Get a page by ID."""
        if not self.client:
            return None

        try:
            page = await self.client.pages.retrieve(page_id)
            return self._parse_page(page)
        except Exception as e:
            logger.error(f"Notion get page error: {e}")
            return None

    async def create_page(
        self,
        parent_id: str,
        title: str,
        parent_type: str = "database",  # "database" or "page"
        properties: Dict = None,
        content: List[Dict] = None,
        icon: str = None,
        cover: str = None
    ) -> Optional[NotionPage]:
        """
        Create a new page.

        Args:
            parent_id: Parent database or page ID
            title: Page title
            parent_type: "database" or "page"
            properties: Additional properties (for database pages)
            content: List of block content to add
            icon: Emoji or URL for icon
            cover: URL for cover image
        """
        if not self.client:
            return None

        try:
            # Build parent
            if parent_type == "database":
                parent = {"database_id": parent_id}
            else:
                parent = {"page_id": parent_id}

            # Build properties
            if parent_type == "database":
                props = properties or {}
                props["Name"] = {"title": [{"text": {"content": title}}]}
            else:
                props = {"title": {"title": [{"text": {"content": title}}]}}

            # Build request
            request = {"parent": parent, "properties": props}

            if icon:
                if icon.startswith("http"):
                    request["icon"] = {"type": "external", "external": {"url": icon}}
                else:
                    request["icon"] = {"type": "emoji", "emoji": icon}

            if cover:
                request["cover"] = {"type": "external", "external": {"url": cover}}

            if content:
                request["children"] = content

            page = await self.client.pages.create(**request)
            return self._parse_page(page)
        except Exception as e:
            logger.error(f"Notion create page error: {e}")
            return None

    async def update_page(
        self,
        page_id: str,
        properties: Dict = None,
        archived: bool = None,
        icon: str = None,
        cover: str = None
    ) -> Optional[NotionPage]:
        """Update a page's properties."""
        if not self.client:
            return None

        try:
            request = {}

            if properties:
                request["properties"] = properties
            if archived is not None:
                request["archived"] = archived
            if icon:
                if icon.startswith("http"):
                    request["icon"] = {"type": "external", "external": {"url": icon}}
                else:
                    request["icon"] = {"type": "emoji", "emoji": icon}
            if cover:
                request["cover"] = {"type": "external", "external": {"url": cover}}

            page = await self.client.pages.update(page_id, **request)
            return self._parse_page(page)
        except Exception as e:
            logger.error(f"Notion update page error: {e}")
            return None

    async def archive_page(self, page_id: str) -> bool:
        """Archive (soft delete) a page."""
        result = await self.update_page(page_id, archived=True)
        return result is not None

    async def restore_page(self, page_id: str) -> bool:
        """Restore an archived page."""
        result = await self.update_page(page_id, archived=False)
        return result is not None

    # ==================== BLOCKS ====================

    async def get_block_children(self, block_id: str, page_size: int = 100) -> List[Dict]:
        """Get children blocks of a block/page."""
        if not self.client:
            return []

        try:
            results = await self.client.blocks.children.list(
                block_id=block_id,
                page_size=min(page_size, 100)
            )
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Notion get blocks error: {e}")
            return []

    async def append_blocks(self, parent_id: str, children: List[Dict]) -> List[Dict]:
        """Append blocks to a page or block."""
        if not self.client:
            return []

        try:
            result = await self.client.blocks.children.append(
                block_id=parent_id,
                children=children
            )
            return result.get("results", [])
        except Exception as e:
            logger.error(f"Notion append blocks error: {e}")
            return []

    async def update_block(self, block_id: str, content: Dict) -> Optional[Dict]:
        """Update a block's content."""
        if not self.client:
            return None

        try:
            return await self.client.blocks.update(block_id, **content)
        except Exception as e:
            logger.error(f"Notion update block error: {e}")
            return None

    async def delete_block(self, block_id: str) -> bool:
        """Delete a block."""
        if not self.client:
            return False

        try:
            await self.client.blocks.delete(block_id)
            return True
        except Exception as e:
            logger.error(f"Notion delete block error: {e}")
            return False

    # ==================== BLOCK BUILDERS ====================

    @staticmethod
    def build_paragraph(text: str, bold: bool = False, italic: bool = False) -> Dict:
        """Build a paragraph block."""
        annotations = {}
        if bold:
            annotations["bold"] = True
        if italic:
            annotations["italic"] = True

        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": text},
                    "annotations": annotations if annotations else None
                }]
            }
        }

    @staticmethod
    def build_heading(text: str, level: int = 1) -> Dict:
        """Build a heading block (level 1-3)."""
        heading_type = f"heading_{min(max(level, 1), 3)}"
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def build_todo(text: str, checked: bool = False) -> Dict:
        """Build a to-do block."""
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "checked": checked
            }
        }

    @staticmethod
    def build_bullet(text: str) -> Dict:
        """Build a bulleted list item."""
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def build_code(code: str, language: str = "python") -> Dict:
        """Build a code block."""
        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": code}}],
                "language": language
            }
        }

    @staticmethod
    def build_callout(text: str, emoji: str = "💡") -> Dict:
        """Build a callout block."""
        return {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "icon": {"type": "emoji", "emoji": emoji}
            }
        }

    @staticmethod
    def build_divider() -> Dict:
        """Build a divider block."""
        return {"object": "block", "type": "divider", "divider": {}}

    @staticmethod
    def build_bookmark(url: str) -> Dict:
        """Build a bookmark block."""
        return {
            "object": "block",
            "type": "bookmark",
            "bookmark": {"url": url}
        }

    # ==================== COMMENTS ====================

    async def get_comments(self, block_id: str = None, page_id: str = None) -> List[Dict]:
        """Get comments on a block or page."""
        if not self.client:
            return []

        try:
            params = {}
            if block_id:
                params["block_id"] = block_id
            elif page_id:
                params["block_id"] = page_id  # Page is a block

            results = await self.client.comments.list(**params)
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Notion get comments error: {e}")
            return []

    async def add_comment(self, page_id: str, text: str) -> Optional[Dict]:
        """Add a comment to a page."""
        if not self.client:
            return None

        try:
            return await self.client.comments.create(
                parent={"page_id": page_id},
                rich_text=[{"type": "text", "text": {"content": text}}]
            )
        except Exception as e:
            logger.error(f"Notion add comment error: {e}")
            return None

    # ==================== USERS ====================

    async def list_users(self) -> List[Dict]:
        """List all users in the workspace."""
        if not self.client:
            return []

        try:
            results = await self.client.users.list()
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Notion list users error: {e}")
            return []

    async def get_user(self, user_id: str) -> Optional[Dict]:
        """Get a specific user."""
        if not self.client:
            return None

        try:
            return await self.client.users.retrieve(user_id)
        except Exception as e:
            logger.error(f"Notion get user error: {e}")
            return None

    async def get_bot_user(self) -> Optional[Dict]:
        """Get the bot user (integration user)."""
        if not self.client:
            return None

        try:
            return await self.client.users.me()
        except Exception as e:
            logger.error(f"Notion get bot user error: {e}")
            return None

    # ==================== UTILITIES ====================

    def _parse_page(self, page: Dict) -> NotionPage:
        """Parse a raw page response into NotionPage."""
        # Extract title
        title = ""
        props = page.get("properties", {})
        for prop_name, prop_data in props.items():
            if prop_data.get("type") == "title":
                title_parts = prop_data.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts)
                break

        # Extract parent info
        parent = page.get("parent", {})
        parent_type = parent.get("type", "")
        parent_id = parent.get(f"{parent_type}_id", "") if parent_type else ""

        # Extract icon
        icon = None
        if page.get("icon"):
            if page["icon"].get("type") == "emoji":
                icon = page["icon"].get("emoji")
            elif page["icon"].get("type") == "external":
                icon = page["icon"]["external"].get("url")

        # Extract cover
        cover = None
        if page.get("cover"):
            if page["cover"].get("type") == "external":
                cover = page["cover"]["external"].get("url")

        return NotionPage(
            id=page["id"],
            title=title,
            url=page.get("url", ""),
            parent_id=parent_id,
            parent_type=parent_type.replace("_id", ""),
            icon=icon,
            cover=cover,
            properties=props,
            created_time=datetime.fromisoformat(page["created_time"].replace("Z", "+00:00")) if page.get("created_time") else None,
            last_edited_time=datetime.fromisoformat(page["last_edited_time"].replace("Z", "+00:00")) if page.get("last_edited_time") else None,
            archived=page.get("archived", False)
        )

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute an action (legacy interface)."""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Notion not connected")

        action_map = {
            "search": lambda p: self.search(p.get("query", "")),
            "create_page": lambda p: self.create_page(
                p.get("parent_id"), p.get("title"), p.get("parent_type", "database"),
                p.get("properties"), p.get("content"), p.get("icon"), p.get("cover")
            ),
            "get_page": lambda p: self.get_page(p.get("page_id")),
            "update_page": lambda p: self.update_page(p.get("page_id"), p.get("properties")),
            "archive_page": lambda p: self.archive_page(p.get("page_id")),
            "query_database": lambda p: self.query_database(p.get("database_id"), p.get("filter"), p.get("sorts")),
            "list_databases": lambda p: self.list_databases(),
            "get_blocks": lambda p: self.get_block_children(p.get("block_id")),
            "append_blocks": lambda p: self.append_blocks(p.get("parent_id"), p.get("children")),
            "add_comment": lambda p: self.add_comment(p.get("page_id"), p.get("text")),
        }

        if action in action_map:
            return await action_map[action](params)
        else:
            raise ValueError(f"Unknown action: {action}")


# ==================== SKILL INTERFACE ====================

class NotionSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self, token: str = None):
        import os
        self.provider = NotionProvider(token or os.environ.get("NOTION_TOKEN", ""))
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Notion."""
        self._connected = await self.provider.connect()
        return self._connected

    async def search(self, query: str) -> List[Dict]:
        """Search workspace."""
        if not self._connected:
            await self.connect()
        results = await self.provider.search(query)
        return results

    async def create_page(self, parent_id: str, title: str, content: str = None) -> Dict:
        """Create a new page."""
        if not self._connected:
            await self.connect()

        blocks = None
        if content:
            blocks = [NotionProvider.build_paragraph(content)]

        page = await self.provider.create_page(
            parent_id=parent_id,
            title=title,
            parent_type="page",
            content=blocks
        )
        return page.to_dict() if page else {}

    async def get_databases(self) -> List[Dict]:
        """List all databases."""
        if not self._connected:
            await self.connect()
        databases = await self.provider.list_databases()
        return [db.to_dict() for db in databases]

    async def query_database(self, database_id: str, filter: Dict = None) -> List[Dict]:
        """Query a database."""
        if not self._connected:
            await self.connect()
        pages = await self.provider.query_database(database_id, filter)
        return [p.to_dict() for p in pages]

    async def add_todo(self, page_id: str, text: str) -> bool:
        """Add a to-do item to a page."""
        if not self._connected:
            await self.connect()
        blocks = await self.provider.append_blocks(
            page_id,
            [NotionProvider.build_todo(text)]
        )
        return len(blocks) > 0


# Global instance (lazy initialization)
notion_skill: Optional[NotionSkill] = None


def get_notion_skill(token: str = None) -> NotionSkill:
    """Get or create the Notion skill instance."""
    global notion_skill
    if notion_skill is None:
        notion_skill = NotionSkill(token)
    return notion_skill
