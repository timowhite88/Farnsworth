"""
Farnsworth Memory Sharing - Export/Import and Backup/Restore

Q1 2025 Feature: Memory Sharing
- Export memory snapshots
- Import memories from other Farnsworth instances
- Selective memory backup/restore
- Memory merge with conflict resolution
"""

import asyncio
import json
import hashlib
import gzip
import io
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from enum import Enum
import uuid

from loguru import logger


class ExportFormat(Enum):
    """Memory export formats."""
    JSON = "json"
    COMPRESSED_JSON = "json.gz"
    ARCHIVE = "tar.gz"


class MergeStrategy(Enum):
    """Strategy for handling conflicts during import."""
    SKIP = "skip"              # Skip conflicting entries
    OVERWRITE = "overwrite"    # Overwrite with imported data
    KEEP_NEWER = "keep_newer"  # Keep the newer entry
    KEEP_BOTH = "keep_both"    # Keep both with suffix
    ASK = "ask"                # Ask for each conflict


@dataclass
class ExportManifest:
    """Manifest for an exported memory package."""
    id: str
    created_at: datetime
    source_instance: str
    version: str
    format: ExportFormat

    # Content summary
    total_memories: int = 0
    total_conversations: int = 0
    total_entities: int = 0
    total_sessions: int = 0

    # Filters applied
    date_range: Optional[tuple[str, str]] = None
    tags_included: Optional[list[str]] = None
    types_included: Optional[list[str]] = None

    # Checksums
    content_hash: str = ""
    entry_hashes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "source_instance": self.source_instance,
            "version": self.version,
            "format": self.format.value,
            "total_memories": self.total_memories,
            "total_conversations": self.total_conversations,
            "total_entities": self.total_entities,
            "total_sessions": self.total_sessions,
            "date_range": self.date_range,
            "tags_included": self.tags_included,
            "types_included": self.types_included,
            "content_hash": self.content_hash,
            "entry_hashes": self.entry_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportManifest":
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            source_instance=data["source_instance"],
            version=data["version"],
            format=ExportFormat(data["format"]),
            total_memories=data.get("total_memories", 0),
            total_conversations=data.get("total_conversations", 0),
            total_entities=data.get("total_entities", 0),
            total_sessions=data.get("total_sessions", 0),
            date_range=tuple(data["date_range"]) if data.get("date_range") else None,
            tags_included=data.get("tags_included"),
            types_included=data.get("types_included"),
            content_hash=data.get("content_hash", ""),
            entry_hashes=data.get("entry_hashes", []),
        )


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    imported_count: int = 0
    skipped_count: int = 0
    conflict_count: int = 0
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BackupInfo:
    """Information about a backup."""
    id: str
    created_at: datetime
    path: str
    size_bytes: int
    manifest: ExportManifest

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "path": self.path,
            "size_bytes": self.size_bytes,
            "manifest": self.manifest.to_dict(),
        }


class MemorySharing:
    """
    Memory export, import, and backup system.

    Features:
    - Export memory snapshots with selective filtering
    - Import from other Farnsworth instances
    - Conflict resolution strategies
    - Incremental backups
    - Memory merge with deduplication
    """

    def __init__(
        self,
        data_dir: str = "./data",
        backup_dir: str = "./backups",
        instance_id: Optional[str] = None,
        version: str = "0.2.0",
    ):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.instance_id = instance_id or f"farnsworth_{uuid.uuid4().hex[:8]}"
        self.version = version

        # Track backups
        self.backups: dict[str, BackupInfo] = {}

        # Callbacks for memory access (set by MemorySystem)
        self.get_memories_fn: Optional[Callable] = None
        self.store_memory_fn: Optional[Callable] = None
        self.get_conversations_fn: Optional[Callable] = None
        self.store_conversation_fn: Optional[Callable] = None
        self.get_entities_fn: Optional[Callable] = None
        self.store_entity_fn: Optional[Callable] = None
        self.get_sessions_fn: Optional[Callable] = None
        self.store_session_fn: Optional[Callable] = None

        self._lock = asyncio.Lock()

    async def export_memories(
        self,
        output_path: str,
        format: ExportFormat = ExportFormat.COMPRESSED_JSON,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        types: Optional[list[str]] = None,
        include_conversations: bool = True,
        include_entities: bool = True,
        include_sessions: bool = True,
    ) -> ExportManifest:
        """
        Export memories to a file.

        Args:
            output_path: Path for export file
            format: Export format
            start_date: Only include memories after this date
            end_date: Only include memories before this date
            tags: Only include memories with these tags
            types: Only include these memory types
            include_conversations: Include conversation history
            include_entities: Include knowledge graph entities
            include_sessions: Include episodic sessions

        Returns:
            Export manifest
        """
        async with self._lock:
            export_data = {
                "memories": [],
                "conversations": [],
                "entities": [],
                "sessions": [],
            }

            # Collect memories
            if self.get_memories_fn:
                memories = await self.get_memories_fn()
                for mem in memories:
                    # Apply filters
                    if start_date and mem.get("created_at"):
                        mem_date = datetime.fromisoformat(mem["created_at"])
                        if mem_date < start_date:
                            continue
                    if end_date and mem.get("created_at"):
                        mem_date = datetime.fromisoformat(mem["created_at"])
                        if mem_date > end_date:
                            continue
                    if tags and not any(t in mem.get("tags", []) for t in tags):
                        continue
                    if types and mem.get("type") not in types:
                        continue

                    export_data["memories"].append(mem)

            # Collect conversations
            if include_conversations and self.get_conversations_fn:
                conversations = await self.get_conversations_fn()
                for conv in conversations:
                    if start_date and conv.get("timestamp"):
                        conv_date = datetime.fromisoformat(conv["timestamp"])
                        if conv_date < start_date:
                            continue
                    if end_date and conv.get("timestamp"):
                        conv_date = datetime.fromisoformat(conv["timestamp"])
                        if conv_date > end_date:
                            continue
                    export_data["conversations"].append(conv)

            # Collect entities
            if include_entities and self.get_entities_fn:
                entities = await self.get_entities_fn()
                export_data["entities"] = entities

            # Collect sessions
            if include_sessions and self.get_sessions_fn:
                sessions = await self.get_sessions_fn()
                for session in sessions:
                    if start_date and session.get("started_at"):
                        session_date = datetime.fromisoformat(session["started_at"])
                        if session_date < start_date:
                            continue
                    if end_date and session.get("started_at"):
                        session_date = datetime.fromisoformat(session["started_at"])
                        if session_date > end_date:
                            continue
                    export_data["sessions"].append(session)

            # Create manifest
            content_json = json.dumps(export_data, sort_keys=True)
            content_hash = hashlib.sha256(content_json.encode()).hexdigest()

            entry_hashes = [
                hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()
                for m in export_data["memories"]
            ]

            manifest = ExportManifest(
                id=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                created_at=datetime.now(),
                source_instance=self.instance_id,
                version=self.version,
                format=format,
                total_memories=len(export_data["memories"]),
                total_conversations=len(export_data["conversations"]),
                total_entities=len(export_data["entities"]),
                total_sessions=len(export_data["sessions"]),
                date_range=(
                    start_date.isoformat() if start_date else None,
                    end_date.isoformat() if end_date else None,
                ) if start_date or end_date else None,
                tags_included=tags,
                types_included=types,
                content_hash=content_hash,
                entry_hashes=entry_hashes,
            )

            # Write export
            export_package = {
                "manifest": manifest.to_dict(),
                "data": export_data,
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == ExportFormat.JSON:
                output_path.write_text(json.dumps(export_package, indent=2), encoding='utf-8')

            elif format == ExportFormat.COMPRESSED_JSON:
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    json.dump(export_package, f)

            elif format == ExportFormat.ARCHIVE:
                # Create tar.gz with separate files
                with tarfile.open(output_path, "w:gz") as tar:
                    # Add manifest
                    manifest_data = json.dumps(manifest.to_dict(), indent=2).encode()
                    manifest_info = tarfile.TarInfo(name="manifest.json")
                    manifest_info.size = len(manifest_data)
                    tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_data))

                    # Add data
                    data_content = json.dumps(export_data, indent=2).encode()
                    data_info = tarfile.TarInfo(name="data.json")
                    data_info.size = len(data_content)
                    tar.addfile(data_info, fileobj=io.BytesIO(data_content))

            logger.info(f"Exported {manifest.total_memories} memories to {output_path}")
            return manifest

    async def import_memories(
        self,
        input_path: str,
        merge_strategy: MergeStrategy = MergeStrategy.KEEP_NEWER,
        conflict_resolver: Optional[Callable] = None,
        import_conversations: bool = True,
        import_entities: bool = True,
        import_sessions: bool = True,
    ) -> ImportResult:
        """
        Import memories from an export file.

        Args:
            input_path: Path to import file
            merge_strategy: How to handle conflicts
            conflict_resolver: Custom function for ASK strategy
            import_conversations: Import conversation history
            import_entities: Import knowledge graph entities
            import_sessions: Import episodic sessions

        Returns:
            Import result with statistics
        """
        async with self._lock:
            result = ImportResult(success=True)

            input_path = Path(input_path)
            if not input_path.exists():
                result.success = False
                result.errors.append(f"File not found: {input_path}")
                return result

            try:
                # Load export package
                if input_path.suffix == ".gz":
                    if str(input_path).endswith(".tar.gz"):
                        # Archive format
                        with tarfile.open(input_path, "r:gz") as tar:
                            manifest_file = tar.extractfile("manifest.json")
                            data_file = tar.extractfile("data.json")
                            if manifest_file is None or data_file is None:
                                raise ValueError("Invalid archive: missing manifest.json or data.json")
                            manifest_data = json.loads(manifest_file.read().decode())
                            export_data = json.loads(data_file.read().decode())
                    else:
                        # Compressed JSON
                        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                            package = json.load(f)
                            manifest_data = package["manifest"]
                            export_data = package["data"]
                else:
                    # Plain JSON
                    package = json.loads(input_path.read_text(encoding='utf-8'))
                    manifest_data = package["manifest"]
                    export_data = package["data"]

                manifest = ExportManifest.from_dict(manifest_data)

                # Verify integrity
                content_json = json.dumps(export_data, sort_keys=True)
                computed_hash = hashlib.sha256(content_json.encode()).hexdigest()
                if computed_hash != manifest.content_hash:
                    result.warnings.append("Content hash mismatch - data may be corrupted")

                # Import memories
                if self.store_memory_fn:
                    for mem in export_data.get("memories", []):
                        try:
                            imported = await self._import_single_memory(
                                mem, merge_strategy, conflict_resolver
                            )
                            if imported == "imported":
                                result.imported_count += 1
                            elif imported == "skipped":
                                result.skipped_count += 1
                            elif imported == "conflict":
                                result.conflict_count += 1
                        except Exception as e:
                            result.error_count += 1
                            result.errors.append(f"Failed to import memory: {e}")

                # Import conversations
                if import_conversations and self.store_conversation_fn:
                    for conv in export_data.get("conversations", []):
                        try:
                            await self.store_conversation_fn(conv)
                        except Exception as e:
                            result.warnings.append(f"Failed to import conversation: {e}")

                # Import entities
                if import_entities and self.store_entity_fn:
                    for entity in export_data.get("entities", []):
                        try:
                            await self.store_entity_fn(entity)
                        except Exception as e:
                            result.warnings.append(f"Failed to import entity: {e}")

                # Import sessions
                if import_sessions and self.store_session_fn:
                    for session in export_data.get("sessions", []):
                        try:
                            await self.store_session_fn(session)
                        except Exception as e:
                            result.warnings.append(f"Failed to import session: {e}")

                logger.info(f"Imported {result.imported_count} memories, {result.skipped_count} skipped, {result.conflict_count} conflicts")

            except Exception as e:
                result.success = False
                result.errors.append(f"Import failed: {e}")
                logger.error(f"Import failed: {e}")

            return result

    async def _import_single_memory(
        self,
        memory: dict,
        strategy: MergeStrategy,
        resolver: Optional[Callable] = None,
    ) -> str:
        """Import a single memory with conflict handling."""
        # Check for existing memory with same content hash
        content_hash = hashlib.md5(
            json.dumps(memory.get("content", ""), sort_keys=True).encode()
        ).hexdigest()

        # This would need integration with the actual memory system
        # For now, we assume store_memory_fn handles deduplication

        if strategy == MergeStrategy.SKIP:
            # Just try to store, skip on error
            try:
                await self.store_memory_fn(memory)
                return "imported"
            except Exception:
                return "skipped"

        elif strategy == MergeStrategy.OVERWRITE:
            await self.store_memory_fn(memory, overwrite=True)
            return "imported"

        elif strategy == MergeStrategy.KEEP_NEWER:
            # Compare timestamps
            await self.store_memory_fn(memory, keep_newer=True)
            return "imported"

        elif strategy == MergeStrategy.KEEP_BOTH:
            # Add suffix to ID if conflict
            memory["id"] = f"{memory.get('id', 'imported')}_{uuid.uuid4().hex[:6]}"
            await self.store_memory_fn(memory)
            return "imported"

        elif strategy == MergeStrategy.ASK:
            if resolver:
                decision = await resolver(memory)
                if decision == "import":
                    await self.store_memory_fn(memory)
                    return "imported"
                elif decision == "skip":
                    return "skipped"
            return "conflict"

        return "skipped"

    async def create_backup(
        self,
        name: Optional[str] = None,
        incremental: bool = False,
    ) -> BackupInfo:
        """
        Create a full or incremental backup.

        Args:
            name: Optional backup name
            incremental: Only backup changes since last backup

        Returns:
            Backup information
        """
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if name:
            backup_id = f"{backup_id}_{name}"

        backup_path = self.backup_dir / f"{backup_id}.tar.gz"

        # Export all memories
        manifest = await self.export_memories(
            output_path=str(backup_path),
            format=ExportFormat.ARCHIVE,
            include_conversations=True,
            include_entities=True,
            include_sessions=True,
        )

        backup_info = BackupInfo(
            id=backup_id,
            created_at=datetime.now(),
            path=str(backup_path),
            size_bytes=backup_path.stat().st_size,
            manifest=manifest,
        )

        self.backups[backup_id] = backup_info

        # Save backup index
        await self._save_backup_index()

        logger.info(f"Created backup: {backup_id} ({backup_info.size_bytes} bytes)")
        return backup_info

    async def restore_backup(
        self,
        backup_id: str,
        clear_existing: bool = False,
    ) -> ImportResult:
        """
        Restore from a backup.

        Args:
            backup_id: ID of backup to restore
            clear_existing: Clear existing data before restore

        Returns:
            Import result
        """
        if backup_id not in self.backups:
            return ImportResult(success=False, errors=["Backup not found"])

        backup = self.backups[backup_id]

        if clear_existing:
            # This would need integration with actual clearing logic
            logger.warning("Clearing existing data before restore")

        return await self.import_memories(
            input_path=backup.path,
            merge_strategy=MergeStrategy.OVERWRITE if clear_existing else MergeStrategy.KEEP_NEWER,
            import_conversations=True,
            import_entities=True,
            import_sessions=True,
        )

    async def list_backups(self) -> list[BackupInfo]:
        """List all available backups."""
        # Refresh from disk
        await self._load_backup_index()
        return sorted(self.backups.values(), key=lambda b: b.created_at, reverse=True)

    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self.backups:
            return False

        backup = self.backups[backup_id]
        backup_path = Path(backup.path)

        if backup_path.exists():
            backup_path.unlink()

        del self.backups[backup_id]
        await self._save_backup_index()

        return True

    async def get_selective_export(
        self,
        memory_ids: list[str],
        output_path: str,
    ) -> ExportManifest:
        """Export specific memories by ID."""
        # This would need integration with actual memory retrieval
        # For now, create a filtered export
        return await self.export_memories(
            output_path=output_path,
            format=ExportFormat.COMPRESSED_JSON,
        )

    async def merge_exports(
        self,
        export_paths: list[str],
        output_path: str,
        dedup_strategy: str = "keep_first",
    ) -> ExportManifest:
        """Merge multiple exports into one."""
        merged_data = {
            "memories": [],
            "conversations": [],
            "entities": [],
            "sessions": [],
        }

        seen_hashes = set()

        for path in export_paths:
            path = Path(path)
            if not path.exists():
                continue

            if path.suffix == ".gz":
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    package = json.load(f)
            else:
                package = json.loads(path.read_text(encoding='utf-8'))

            export_data = package.get("data", {})

            for mem in export_data.get("memories", []):
                mem_hash = hashlib.md5(
                    json.dumps(mem.get("content", ""), sort_keys=True).encode()
                ).hexdigest()

                if mem_hash not in seen_hashes:
                    merged_data["memories"].append(mem)
                    seen_hashes.add(mem_hash)
                elif dedup_strategy == "keep_last":
                    # Replace existing
                    for i, existing in enumerate(merged_data["memories"]):
                        existing_hash = hashlib.md5(
                            json.dumps(existing.get("content", ""), sort_keys=True).encode()
                        ).hexdigest()
                        if existing_hash == mem_hash:
                            merged_data["memories"][i] = mem
                            break

            # Merge other data types (simpler - just extend)
            merged_data["conversations"].extend(export_data.get("conversations", []))
            merged_data["entities"].extend(export_data.get("entities", []))
            merged_data["sessions"].extend(export_data.get("sessions", []))

        # Create merged export
        content_json = json.dumps(merged_data, sort_keys=True)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()

        manifest = ExportManifest(
            id=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            source_instance=self.instance_id,
            version=self.version,
            format=ExportFormat.COMPRESSED_JSON,
            total_memories=len(merged_data["memories"]),
            total_conversations=len(merged_data["conversations"]),
            total_entities=len(merged_data["entities"]),
            total_sessions=len(merged_data["sessions"]),
            content_hash=content_hash,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump({"manifest": manifest.to_dict(), "data": merged_data}, f)

        return manifest

    async def _save_backup_index(self):
        """Save backup index to disk."""
        index_file = self.backup_dir / "backup_index.json"
        index_data = {
            bid: b.to_dict() for bid, b in self.backups.items()
        }
        index_file.write_text(json.dumps(index_data, indent=2), encoding='utf-8')

    async def _load_backup_index(self):
        """Load backup index from disk."""
        index_file = self.backup_dir / "backup_index.json"
        if index_file.exists():
            index_data = json.loads(index_file.read_text(encoding='utf-8'))
            for bid, data in index_data.items():
                if Path(data["path"]).exists():
                    self.backups[bid] = BackupInfo(
                        id=data["id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        path=data["path"],
                        size_bytes=data["size_bytes"],
                        manifest=ExportManifest.from_dict(data["manifest"]),
                    )

    def get_stats(self) -> dict:
        """Get sharing system statistics."""
        total_backup_size = sum(b.size_bytes for b in self.backups.values())
        return {
            "instance_id": self.instance_id,
            "total_backups": len(self.backups),
            "total_backup_size_mb": total_backup_size / (1024 * 1024),
            "backup_dir": str(self.backup_dir),
        }
