"""
Protected Chain Memory Core

This module contains the core upload/download logic for on-chain memory storage.
The source is compiled to a binary for distribution - source should NOT be public.

FARNS TOKEN REQUIRED: 100,000+ FARNS on Solana to use this feature.
Token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

To compile for distribution:
    python scripts/build_protected.py --method cython

Copyright (c) 2026 Farnsworth AI. All rights reserved.
"""

from .core import (
    ChainUploader,
    ChainDownloader,
    verify_installation,
    get_fingerprint,
    verify_farns_for_operation,
    MIN_FARNS_REQUIRED,
    FARNS_TOKEN_MINT,
)

__all__ = [
    "ChainUploader",
    "ChainDownloader",
    "verify_installation",
    "get_fingerprint",
    "verify_farns_for_operation",
    "MIN_FARNS_REQUIRED",
    "FARNS_TOKEN_MINT",
]

# Note: Verification happens on each upload/download operation now
# No longer verify on import to avoid blocking non-chain-memory usage
