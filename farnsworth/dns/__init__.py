"""
Farnsworth DNS Management Package

"Good news, everyone! I can now control the very fabric of the internet!"

Multi-provider DNS management including MX records and SSL certificates.
"""

from farnsworth.dns.dns_manager import (
    DNSManager,
    DNSRecord,
    DNSRecordType,
)
from farnsworth.dns.ssl_certificates import (
    SSLManager,
    Certificate,
)

__all__ = [
    "DNSManager",
    "DNSRecord",
    "DNSRecordType",
    "SSLManager",
    "Certificate",
]
