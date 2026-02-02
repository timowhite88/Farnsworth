#!/usr/bin/env python3
"""
Build Protected Chain Memory Module

Compiles the protected core.py to a binary that can be distributed
without revealing the source code.

Usage:
    python scripts/build_protected.py [--method cython|nuitka|pyarmor]

The compiled binary will be placed in:
    farnsworth/integration/chain_memory/protected/

After building:
    1. The source core.py is gitignored (not committed)
    2. Only the compiled .pyd/.so is distributed
    3. Users need 100k+ FARNS to use the tool

Copyright (c) 2026 Farnsworth AI. All rights reserved.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROTECTED_DIR = PROJECT_ROOT / "farnsworth" / "integration" / "chain_memory" / "protected"
SOURCE_FILE = PROTECTED_DIR / "core.py"


def check_source_exists():
    """Verify source file exists."""
    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        print("\nThe source file may have already been removed after compilation.")
        print("If you need to rebuild, restore core.py from a secure backup.")
        sys.exit(1)


def build_with_cython():
    """Build using Cython (recommended)."""
    print("Building with Cython...")

    try:
        import Cython
        print(f"Cython version: {Cython.__version__}")
    except ImportError:
        print("Installing Cython...")
        subprocess.run([sys.executable, "-m", "pip", "install", "cython"], check=True)

    # Create setup.py for Cython
    setup_py = PROTECTED_DIR / "setup_cython.py"
    setup_content = '''
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "core.py",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        }
    )
)
'''

    with open(setup_py, 'w') as f:
        f.write(setup_content)

    # Build
    os.chdir(PROTECTED_DIR)
    result = subprocess.run(
        [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True
    )

    # Cleanup
    setup_py.unlink(missing_ok=True)
    shutil.rmtree(PROTECTED_DIR / "build", ignore_errors=True)

    if result.returncode != 0:
        print(f"Cython build failed:\n{result.stderr}")
        sys.exit(1)

    # Find the built file
    for f in PROTECTED_DIR.glob("core*.so"):
        print(f"Built: {f}")
        return f
    for f in PROTECTED_DIR.glob("core*.pyd"):
        print(f"Built: {f}")
        return f

    print("Build completed but no output file found!")
    return None


def build_with_nuitka():
    """Build using Nuitka (creates native binary)."""
    print("Building with Nuitka...")

    try:
        import nuitka
    except ImportError:
        print("Installing Nuitka...")
        subprocess.run([sys.executable, "-m", "pip", "install", "nuitka"], check=True)

    os.chdir(PROTECTED_DIR)
    result = subprocess.run(
        [sys.executable, "-m", "nuitka", "--module", "--remove-output", "core.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Nuitka build failed:\n{result.stderr}")
        sys.exit(1)

    print("Nuitka build completed")
    return PROTECTED_DIR / "core.pyd" if sys.platform == "win32" else PROTECTED_DIR / "core.so"


def build_with_pyarmor():
    """Build using PyArmor (obfuscation)."""
    print("Building with PyArmor...")

    try:
        import pyarmor
    except ImportError:
        print("Installing PyArmor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarmor"], check=True)

    os.chdir(PROTECTED_DIR)

    # PyArmor obfuscates but doesn't compile to binary
    # The result is still Python but obfuscated
    result = subprocess.run(
        [sys.executable, "-m", "pyarmor", "gen", "-O", "dist", "core.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"PyArmor build failed:\n{result.stderr}")
        sys.exit(1)

    # Move obfuscated file
    obfuscated = PROTECTED_DIR / "dist" / "core.py"
    if obfuscated.exists():
        # Rename original
        SOURCE_FILE.rename(SOURCE_FILE.with_suffix('.py.bak'))
        # Move obfuscated to main location
        shutil.copy(obfuscated, SOURCE_FILE)
        shutil.rmtree(PROTECTED_DIR / "dist", ignore_errors=True)
        print(f"Obfuscated: {SOURCE_FILE}")
        print("Note: PyArmor creates obfuscated Python, not a compiled binary")
        return SOURCE_FILE

    print("PyArmor build completed but no output found")
    return None


def update_init_file():
    """Update __init__.py to try importing compiled module first."""
    init_file = PROTECTED_DIR / "__init__.py"

    new_content = '''"""
Protected Chain Memory Core

This module contains the core upload/download logic for on-chain memory storage.
The source is compiled to a binary for distribution.

FARNS Token Required: 100,000+ FARNS on Solana
Token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS

Copyright (c) 2026 Farnsworth AI. All rights reserved.
"""

import sys
import importlib.util

# Try to import compiled module first, fall back to source for development
_module = None

# Try compiled Cython/Nuitka module
for suffix in ['.cpython-311-x86_64-linux-gnu.so', '.cpython-310-x86_64-linux-gnu.so',
               '.cpython-39-x86_64-linux-gnu.so', '.pyd', '.so']:
    try:
        spec = importlib.util.find_spec(f".core", package=__name__)
        if spec and spec.origin and spec.origin.endswith(suffix):
            _module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_module)
            break
    except:
        pass

# Fall back to source (development only - should not be in production)
if _module is None:
    try:
        from .core import (
            ChainUploader,
            ChainDownloader,
            verify_installation,
            get_fingerprint,
            verify_farns_for_operation,
            MIN_FARNS_REQUIRED,
            FARNS_TOKEN_MINT,
        )
    except ImportError as e:
        raise ImportError(
            "Chain Memory protected module not found.\\n"
            "This module requires compilation for distribution.\\n"
            f"Error: {e}"
        )
else:
    ChainUploader = _module.ChainUploader
    ChainDownloader = _module.ChainDownloader
    verify_installation = _module.verify_installation
    get_fingerprint = _module.get_fingerprint
    verify_farns_for_operation = _module.verify_farns_for_operation
    MIN_FARNS_REQUIRED = _module.MIN_FARNS_REQUIRED
    FARNS_TOKEN_MINT = _module.FARNS_TOKEN_MINT

__all__ = [
    "ChainUploader",
    "ChainDownloader",
    "verify_installation",
    "get_fingerprint",
    "verify_farns_for_operation",
    "MIN_FARNS_REQUIRED",
    "FARNS_TOKEN_MINT",
]
'''

    with open(init_file, 'w') as f:
        f.write(new_content)

    print(f"Updated: {init_file}")


def main():
    parser = argparse.ArgumentParser(description="Build protected Chain Memory module")
    parser.add_argument(
        "--method",
        choices=["cython", "nuitka", "pyarmor"],
        default="cython",
        help="Compilation method (default: cython)"
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep source file after building (for development)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CHAIN MEMORY - Protected Module Builder")
    print("=" * 60)
    print()

    check_source_exists()

    # Build
    if args.method == "cython":
        output = build_with_cython()
    elif args.method == "nuitka":
        output = build_with_nuitka()
    elif args.method == "pyarmor":
        output = build_with_pyarmor()

    if output:
        print(f"\nBuild successful: {output}")

        # Update __init__.py
        update_init_file()

        if not args.keep_source and args.method != "pyarmor":
            # Remove source file (it's gitignored anyway)
            backup = SOURCE_FILE.with_suffix('.py.original')
            SOURCE_FILE.rename(backup)
            print(f"\nSource moved to: {backup}")
            print("(Keep this backup secure - needed for future builds)")

        print("\n" + "=" * 60)
        print("  BUILD COMPLETE")
        print("=" * 60)
        print(f"""
Next steps:
1. The compiled binary is ready for distribution
2. Source file is gitignored (won't be committed)
3. Users need 100k+ FARNS tokens to use Chain Memory

To test:
    python -c "from farnsworth.integration.chain_memory.protected import ChainUploader"

To distribute:
    git add farnsworth/integration/chain_memory/protected/
    git commit -m "feat: Add compiled chain memory module"
    git push
""")
    else:
        print("\nBuild failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
