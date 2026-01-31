"""
Farnsworth Knowledge Base

This module provides access to Farnsworth's knowledge files for the memory system.
Knowledge is loaded into the retrieval database so bots know what capabilities exist.
"""

import os
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent


def get_knowledge_files() -> list:
    """Get all knowledge files in the knowledge directory."""
    files = []
    for f in KNOWLEDGE_DIR.glob("*.md"):
        files.append(f)
    return files


def load_api_knowledge() -> str:
    """Load the API capabilities knowledge base."""
    api_file = KNOWLEDGE_DIR / "api_capabilities.md"
    if api_file.exists():
        return api_file.read_text(encoding="utf-8")
    return ""


def get_all_knowledge() -> dict:
    """Load all knowledge files into a dictionary."""
    knowledge = {}
    for f in get_knowledge_files():
        knowledge[f.stem] = f.read_text(encoding="utf-8")
    return knowledge
