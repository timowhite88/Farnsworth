"""
Farnsworth Sysadmin Tools

"Good news, everyone! I've built a complete system administration toolkit!"

System monitoring, service management, log analysis, network diagnostics, and backups.
"""

try:
    from farnsworth.tools.sysadmin.system_monitor import (
        SystemMonitor,
        SystemMetrics,
        system_monitor,
    )
except ImportError:
    SystemMonitor = SystemMetrics = system_monitor = None

try:
    from farnsworth.tools.sysadmin.service_manager import (
        ServiceManager,
        ServiceInfo,
        ServiceStatus,
        service_manager,
    )
except ImportError:
    ServiceManager = ServiceInfo = ServiceStatus = service_manager = None

try:
    from farnsworth.tools.sysadmin.log_analyzer import (
        LogAnalyzer,
        LogEntry,
        LogLevel,
        LogAnalysisResult,
        log_analyzer,
    )
except ImportError:
    LogAnalyzer = LogEntry = LogLevel = LogAnalysisResult = log_analyzer = None

try:
    from farnsworth.tools.sysadmin.network_tools import (
        NetworkTools,
        HostInfo,
        PortScanResult,
        network_tools,
    )
except ImportError:
    NetworkTools = HostInfo = PortScanResult = network_tools = None

try:
    from farnsworth.tools.sysadmin.backup_manager import (
        BackupManager,
        BackupJob,
        BackupType,
        backup_manager,
    )
except ImportError:
    BackupManager = BackupJob = BackupType = backup_manager = None


__all__ = [
    "SystemMonitor", "SystemMetrics", "system_monitor",
    "ServiceManager", "ServiceInfo", "ServiceStatus", "service_manager",
    "LogAnalyzer", "LogEntry", "LogLevel", "LogAnalysisResult", "log_analyzer",
    "NetworkTools", "HostInfo", "PortScanResult", "network_tools",
    "BackupManager", "BackupJob", "BackupType", "backup_manager",
]
