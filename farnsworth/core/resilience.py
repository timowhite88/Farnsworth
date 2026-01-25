"""
Farnsworth Resilience Module - Reliability & Health Monitoring

Features:
- Automated Backups: Periodic snapshots of the data directory.
- Health Monitoring: Real-time system health checks.
- Graceful Degradation: Fallback strategies when components fail.
"""

import asyncio
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import json
import psutil

from loguru import logger

@dataclass
class HealthStatus:
    status: str  # "healthy", "degraded", "unhealthy"
    components: dict[str, str]
    system_metrics: dict[str, float]
    timestamp: str

class BackupManager:
    """Manages automated backups of system data."""

    def __init__(
        self,
        data_dir: str,
        backup_dir: str,
        retention_count: int = 5,
        interval_hours: float = 24.0
    ):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.retention_count = retention_count
        self.interval_hours = interval_hours
        self._is_running = False
        self._task: Optional[asyncio.Task] = None

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start auto-backup loop."""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._backup_loop())
        logger.info("Backup Manager started")

    async def stop(self):
        """Stop auto-backup loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Backup Manager stopped")

    async def _backup_loop(self):
        while self._is_running:
            try:
                await self.create_backup()
            except Exception as e:
                logger.error(f"Backup failed: {e}")
            
            # Wait for interval
            await asyncio.sleep(self.interval_hours * 3600)

    async def create_backup(self) -> str:
        """Create a snapshot of the data directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        # Use run_in_executor to avoid blocking event loop during file IO
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            shutil.copytree, 
            self.data_dir, 
            backup_path, 
            shutil.ignore_patterns('*.lock', '*.tmp')
        )
        
        logger.info(f"Backup created at {backup_path}")
        
        await self._cleanup_old_backups()
        return str(backup_path)

    async def _cleanup_old_backups(self):
        """Remove old backups exceeding retention count."""
        backups = sorted(self.backup_dir.glob("backup_*"))
        if len(backups) > self.retention_count:
            for old_backup in backups[:-self.retention_count]:
                shutil.rmtree(old_backup)
                logger.info(f"Removed old backup: {old_backup.name}")

    async def restore_backup(self, backup_name: str) -> bool:
        """Restore a specific backup (DANGEROUS: Overwrites data)."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            return False
        
        # Safety backup before restore
        temp_backup = self.data_dir.parent / f"pre_restore_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if self.data_dir.exists():
             shutil.move(str(self.data_dir), str(temp_backup))

        shutil.copytree(backup_path, self.data_dir)
        logger.warning(f"Restored data from {backup_name}")
        return True

class HealthMonitor:
    """Monitors system health and metrics."""

    def __init__(self, check_interval_seconds: float = 60.0):
        self.check_interval = check_interval_seconds
        self._is_running = False
        self.components_to_check: dict[str, Callable] = {}
        self.latest_status: Optional[HealthStatus] = None

    def register_check(self, name: str, check_fn: Callable):
        """Register a component health check function."""
        self.components_to_check[name] = check_fn

    async def check_health(self) -> HealthStatus:
        """Run all health checks immediately."""
        component_status = {}
        info_count = 0
        error_count = 0

        # System Metrics
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        metrics = {
            "cpu_percent": cpu_percent,
            "ram_percent": ram.percent,
            "disk_percent": disk.percent,
            "available_ram_gb": ram.available / (1024**3)
        }

        # Component Checks
        for name, check_fn in self.components_to_check.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    status = await check_fn()
                else:
                    status = check_fn()
                component_status[name] = status
                if status != "healthy":
                    error_count += 1
            except Exception as e:
                component_status[name] = f"error: {str(e)}"
                error_count += 1

        overall = "healthy"
        if error_count > 0:
            overall = "degraded"
        
        if ram.percent > 90 or disk.percent > 95:
            overall = "unhealthy"

        self.latest_status = HealthStatus(
            status=overall,
            components=component_status,
            system_metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        return self.latest_status
