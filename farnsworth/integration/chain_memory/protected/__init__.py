"""
Protected Chain Memory Core

This module contains the core upload/download logic for on-chain memory storage.
In production, this should be compiled with Cython or obfuscated with PyArmor.

DO NOT DISTRIBUTE SOURCE CODE - Use compiled .pyd/.so files only.

To compile for distribution:
    cython -3 core.py
    # OR
    pyarmor gen --pack onefile core.py
"""

from .core import (
    ChainUploader,
    ChainDownloader,
    verify_installation,
    get_fingerprint
)

__all__ = [
    "ChainUploader",
    "ChainDownloader",
    "verify_installation",
    "get_fingerprint"
]

# Verify on import
_VERIFIED = verify_installation()
