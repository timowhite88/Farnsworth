"""
Farnsworth Security Tools

"Good news, everyone! I've built security tools for responsible researchers!"

Comprehensive offensive and defensive security toolkit for authorized testing.
"""

from farnsworth.tools.security.vulnerability_scanner import (
    VulnerabilityScanner,
    VulnerabilityReport,
    vulnerability_scanner,
)
from farnsworth.tools.security.threat_analyzer import (
    ThreatAnalyzer,
    ThreatIndicator,
    threat_analyzer,
)
from farnsworth.tools.security.forensics import (
    ForensicsToolkit,
    forensics_toolkit,
)
from farnsworth.tools.security.header_analyzer import (
    HeaderAnalyzer,
    EmailHeaderAnalysis,
    header_analyzer,
)
from farnsworth.tools.security.recon import (
    ReconEngine,
    ReconResult,
    recon_engine,
)
from farnsworth.tools.security.edr import (
    EDREngine,
    SecurityAlert,
    DetectionRule,
    edr_engine,
)
from farnsworth.tools.security.log_parser import (
    SecurityLogParser,
    ParsedLogEntry,
    LogAnalysisReport,
    security_log_parser,
)


__all__ = [
    # Vulnerability Scanner
    "VulnerabilityScanner",
    "VulnerabilityReport",
    "vulnerability_scanner",
    # Threat Analyzer
    "ThreatAnalyzer",
    "ThreatIndicator",
    "threat_analyzer",
    # Forensics
    "ForensicsToolkit",
    "forensics_toolkit",
    # Header Analyzer
    "HeaderAnalyzer",
    "EmailHeaderAnalysis",
    "header_analyzer",
    # Recon Engine
    "ReconEngine",
    "ReconResult",
    "recon_engine",
    # EDR
    "EDREngine",
    "SecurityAlert",
    "DetectionRule",
    "edr_engine",
    # Log Parser
    "SecurityLogParser",
    "ParsedLogEntry",
    "LogAnalysisReport",
    "security_log_parser",
]
