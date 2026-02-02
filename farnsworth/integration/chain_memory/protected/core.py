"""
Protected Core - On-Chain Memory Upload/Download

This is the protected core logic for uploading/downloading memory to Monad blockchain.
Uses BetterClips-style chunked calldata storage.

PROTECTION NOTICE:
==================
This code should be compiled before distribution:
    - Use Cython: cythonize -i core.py
    - Use PyArmor: pyarmor gen core.py
    - Use Nuitka: nuitka --module core.py

The compiled .pyd/.so file should be distributed instead of this source.

DO NOT REDISTRIBUTE SOURCE CODE WITHOUT AUTHORIZATION.
Copyright (c) 2026 Farnsworth AI. All rights reserved.
"""

import os
import sys
import json
import hashlib
import platform
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger("chain_memory.protected")

# =============================================================================
# PROTECTION & VERIFICATION
# =============================================================================

# Magic bytes for memory chunks on chain
MEMORY_SIGNATURE = b'FARNSMEM'  # 8 bytes
CHUNK_HEADER_SIZE = 44  # signature(8) + memoryId(32) + index(4)

# Monad network configuration
# Uses BetterClips-style raw calldata storage (NO CONTRACT)
# Data is stored in transaction input field, retrievable via eth_getTransactionByHash
#
# Chunk sizes based on BetterClips production usage:
#   - Default: 48-80KB (proven to work reliably)
#   - Private RPC: Can try larger, but 80KB is safe
MONAD_CONFIG = {
    "chain_id": 143,
    "rpc_url": os.getenv("MONAD_RPC", "https://rpc.monad.xyz"),
    "explorer": "https://explorer.monad.xyz",
    # Default 80KB - matches BetterClips production usage
    # Configurable via CHUNK_SIZE_KB env var
    "chunk_size": int(os.getenv("CHUNK_SIZE_KB", "80")) * 1024,
    # BetterClips uses 48-81KB in production, so 80KB is safe default
    "chunk_size_default": 80 * 1024,   # 80KB default (proven)
    "chunk_size_safe": 48 * 1024,      # 48KB conservative
    "chunk_size_large": 128 * 1024,    # 128KB for fast private RPC
    # Transaction settings
    "tx_value": "0.0001",  # Tiny dust amount sent with each chunk (in MON)
    "gas_per_chunk": 100000,
}


def get_fingerprint() -> str:
    """
    Generate a hardware fingerprint for this installation.
    Used to bind the software to specific machines.
    """
    components = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        str(os.getuid()) if hasattr(os, 'getuid') else platform.node(),
    ]

    # Add disk serial if available (Windows)
    if sys.platform == 'win32':
        try:
            import subprocess
            result = subprocess.run(
                ['wmic', 'diskdrive', 'get', 'serialnumber'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                serial = result.stdout.strip().split('\n')[-1].strip()
                if serial:
                    components.append(serial)
        except:
            pass

    fingerprint_data = '|'.join(components)
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]


def verify_installation() -> bool:
    """
    Verify this is a legitimate installation.
    Returns True if valid, False otherwise.
    """
    # In compiled version, this would check license/activation
    # For development, always return True
    fingerprint = get_fingerprint()
    logger.debug(f"Installation fingerprint: {fingerprint}")
    return True


def _check_authorized() -> None:
    """Internal check - raises if not authorized."""
    if not verify_installation():
        raise RuntimeError(
            "Unauthorized installation. Please activate your license at "
            "https://farnsworth.ai/activate"
        )


# =============================================================================
# CHUNK ENCODING
# =============================================================================

def create_chunk_data(
    memory_id: str,
    chunk_index: int,
    chunk_bytes: bytes
) -> bytes:
    """
    Create chunk data with BetterClips-compatible header.

    Format:
        [8 bytes: FARNSMEM signature]
        [32 bytes: memory_id (zero-padded)]
        [4 bytes: chunk index (big-endian)]
        [N bytes: actual data]
    """
    # Signature
    signature = MEMORY_SIGNATURE

    # Memory ID - truncate/pad to 32 bytes
    memory_id_bytes = memory_id.encode('utf-8')[:32]
    memory_id_padded = memory_id_bytes.ljust(32, b'\x00')

    # Chunk index as 4-byte big-endian
    index_bytes = chunk_index.to_bytes(4, byteorder='big')

    # Combine all parts
    return signature + memory_id_padded + index_bytes + chunk_bytes


def parse_chunk_data(data: bytes) -> Tuple[str, int, bytes]:
    """
    Parse chunk data and extract components.

    Returns:
        Tuple of (memory_id, chunk_index, chunk_bytes)
    """
    # Verify signature
    if data[:8] != MEMORY_SIGNATURE:
        # Try BetterClips format
        if data[:8] == b'BCLIPS01':
            # Compatible format
            pass
        else:
            raise ValueError("Invalid chunk signature")

    # Extract memory ID
    memory_id_bytes = data[8:40]
    memory_id = memory_id_bytes.rstrip(b'\x00').decode('utf-8')

    # Extract chunk index
    index_bytes = data[40:44]
    chunk_index = int.from_bytes(index_bytes, byteorder='big')

    # Extract chunk data
    chunk_bytes = data[44:]

    return memory_id, chunk_index, chunk_bytes


# =============================================================================
# CHAIN UPLOADER
# =============================================================================

@dataclass
class UploadResult:
    """Result of a memory upload operation."""
    success: bool
    memory_id: str
    tx_hashes: List[str]
    total_chunks: int
    total_size: int
    total_cost: float
    cost_currency: str
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class ChainUploader:
    """
    Handles uploading memory to Monad blockchain.

    Uses BetterClips-style raw transaction calldata storage.
    NO CONTRACT NEEDED - data stored directly in tx input field.

    How it works:
        1. Split data into chunks (default 80KB, same as BetterClips)
        2. Send each chunk as a transaction TO YOUR OWN ADDRESS
        3. Data is stored in the transaction's 'input' (calldata) field
        4. Retrieve later via eth_getTransactionByHash

    This is permanent, immutable storage on Monad blockchain.
    """

    def __init__(
        self,
        private_key: str,
        rpc_url: Optional[str] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize the uploader.

        Args:
            private_key: Wallet private key (0x prefixed)
            rpc_url: Monad RPC endpoint
            chunk_size: Size of each chunk in bytes (default 80KB)
        """
        _check_authorized()

        self.private_key = private_key
        self.rpc_url = rpc_url or MONAD_CONFIG["rpc_url"]
        self.chunk_size = chunk_size or MONAD_CONFIG["chunk_size"]

        # Initialize web3
        self._init_web3()

    def _init_web3(self):
        """Initialize web3 connection."""
        try:
            from web3 import Web3
            from eth_account import Account

            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address

            if not self.w3.is_connected():
                raise ConnectionError(f"Cannot connect to {self.rpc_url}")

            logger.info(f"Connected to Monad. Wallet: {self.address[:10]}...")

        except ImportError:
            raise RuntimeError("web3 not installed. Run: pip install web3")

    def estimate_cost(self, data_size: int) -> Dict:
        """
        Estimate the cost to upload data of given size.

        Returns:
            Dict with cost breakdown
        """
        num_chunks = (data_size + self.chunk_size - 1) // self.chunk_size
        gas_per_chunk = MONAD_CONFIG["gas_per_chunk"]
        total_gas = num_chunks * gas_per_chunk

        # Get current gas price
        gas_price = self.w3.eth.gas_price
        gas_price_gwei = gas_price / 1e9

        # Calculate cost in MON
        total_cost_wei = total_gas * gas_price
        total_cost_mon = total_cost_wei / 1e18

        # Estimate USD (conservative $0.50/MON)
        mon_price_usd = float(os.getenv("MON_PRICE_USD", "0.50"))
        total_cost_usd = total_cost_mon * mon_price_usd

        return {
            "data_size": data_size,
            "num_chunks": num_chunks,
            "chunk_size": self.chunk_size,
            "gas_per_chunk": gas_per_chunk,
            "total_gas": total_gas,
            "gas_price_gwei": gas_price_gwei,
            "total_cost_mon": total_cost_mon,
            "total_cost_usd": total_cost_usd,
            "mon_price_usd": mon_price_usd
        }

    def chunk_file(self, file_path: str) -> List[bytes]:
        """Split a file into chunks."""
        chunks = []
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        return chunks

    def chunk_bytes(self, data: bytes) -> List[bytes]:
        """Split bytes into chunks."""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunks.append(data[i:i + self.chunk_size])
        return chunks

    async def upload(
        self,
        mp4_path: str,
        index_path: str,
        title: str = "Bot Memory",
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> UploadResult:
        """
        Upload memory MP4 and index to blockchain.

        Args:
            mp4_path: Path to the memvid MP4 file
            index_path: Path to the memvid index JSON
            title: Title for this memory upload
            on_progress: Callback(current_chunk, total_chunks, status)

        Returns:
            UploadResult with transaction details
        """
        import asyncio

        # Generate unique memory ID
        memory_id = hashlib.sha256(
            f"{self.address}:{title}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        logger.info(f"Uploading memory: {memory_id}")

        # Read files
        with open(mp4_path, 'rb') as f:
            mp4_data = f.read()
        with open(index_path, 'r') as f:
            index_data = json.load(f)

        # Create combined payload
        # Format: [4 bytes: index_len][index_json][mp4_data]
        index_bytes = json.dumps(index_data).encode('utf-8')
        index_len = len(index_bytes).to_bytes(4, byteorder='big')
        combined_data = index_len + index_bytes + mp4_data

        # Split into chunks
        chunks = self.chunk_bytes(combined_data)
        total_chunks = len(chunks)

        logger.info(f"Total size: {len(combined_data):,} bytes in {total_chunks} chunks")

        # Estimate cost
        estimate = self.estimate_cost(len(combined_data))
        logger.info(f"Estimated cost: {estimate['total_cost_mon']:.4f} MON (${estimate['total_cost_usd']:.2f})")

        # Upload chunks
        tx_hashes = []
        total_gas_used = 0

        for i, chunk in enumerate(chunks):
            if on_progress:
                on_progress(i + 1, total_chunks, f"Uploading chunk {i + 1}/{total_chunks}")

            # Create chunk with header
            chunk_data = create_chunk_data(memory_id, i, chunk)

            # Send transaction
            try:
                nonce = self.w3.eth.get_transaction_count(self.address)

                tx = {
                    'from': self.address,
                    'to': self.address,  # Send to self (calldata storage)
                    'value': self.w3.to_wei(0.0001, 'ether'),  # Dust amount
                    'data': chunk_data,
                    'gas': MONAD_CONFIG["gas_per_chunk"],
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': nonce,
                    'chainId': MONAD_CONFIG["chain_id"]
                }

                signed = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                tx_hash_hex = tx_hash.hex()

                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

                if receipt['status'] != 1:
                    raise RuntimeError(f"Transaction failed: {tx_hash_hex}")

                tx_hashes.append(tx_hash_hex)
                total_gas_used += receipt['gasUsed']

                logger.debug(f"Chunk {i + 1}/{total_chunks}: {tx_hash_hex}")

            except Exception as e:
                logger.error(f"Failed to upload chunk {i}: {e}")
                return UploadResult(
                    success=False,
                    memory_id=memory_id,
                    tx_hashes=tx_hashes,
                    total_chunks=total_chunks,
                    total_size=len(combined_data),
                    total_cost=0,
                    cost_currency="MON",
                    error=str(e)
                )

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

        # Calculate actual cost
        gas_price = self.w3.eth.gas_price
        actual_cost_wei = total_gas_used * gas_price
        actual_cost_mon = actual_cost_wei / 1e18

        # Create metadata for local storage
        metadata = {
            "memory_id": memory_id,
            "title": title,
            "tx_hashes": tx_hashes,
            "total_chunks": total_chunks,
            "total_size": len(combined_data),
            "mp4_size": len(mp4_data),
            "index_size": len(index_bytes),
            "uploaded_at": datetime.now().isoformat(),
            "wallet": self.address,
            "chain": "monad",
            "chain_id": MONAD_CONFIG["chain_id"]
        }

        if on_progress:
            on_progress(total_chunks, total_chunks, "Upload complete!")

        logger.info(f"Upload complete! Memory ID: {memory_id}")
        logger.info(f"Actual cost: {actual_cost_mon:.4f} MON")

        return UploadResult(
            success=True,
            memory_id=memory_id,
            tx_hashes=tx_hashes,
            total_chunks=total_chunks,
            total_size=len(combined_data),
            total_cost=actual_cost_mon,
            cost_currency="MON",
            metadata=metadata
        )


# =============================================================================
# CHAIN DOWNLOADER
# =============================================================================

@dataclass
class DownloadResult:
    """Result of a memory download operation."""
    success: bool
    memory_id: str
    mp4_path: Optional[str]
    index_path: Optional[str]
    total_chunks: int
    total_size: int
    error: Optional[str] = None


class ChainDownloader:
    """
    Handles downloading memory from Monad blockchain.

    Reconstructs MP4 and index from transaction calldata.
    """

    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            rpc_url: Monad RPC endpoint
        """
        _check_authorized()

        self.rpc_url = rpc_url or MONAD_CONFIG["rpc_url"]
        self._init_web3()

    def _init_web3(self):
        """Initialize web3 connection."""
        try:
            from web3 import Web3
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            if not self.w3.is_connected():
                raise ConnectionError(f"Cannot connect to {self.rpc_url}")

            logger.info("Connected to Monad for download")

        except ImportError:
            raise RuntimeError("web3 not installed. Run: pip install web3")

    def get_tx_data(self, tx_hash: str) -> bytes:
        """Fetch transaction calldata."""
        tx = self.w3.eth.get_transaction(tx_hash)
        return bytes(tx['input'])

    async def download(
        self,
        tx_hashes: List[str],
        output_dir: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> DownloadResult:
        """
        Download and reconstruct memory from transaction hashes.

        Args:
            tx_hashes: List of transaction hashes in order
            output_dir: Directory to save reconstructed files
            on_progress: Callback(current_chunk, total_chunks, status)

        Returns:
            DownloadResult with file paths
        """
        import asyncio

        total_chunks = len(tx_hashes)
        logger.info(f"Downloading {total_chunks} chunks...")

        # Fetch all chunks
        chunks_data = []
        memory_id = None

        for i, tx_hash in enumerate(tx_hashes):
            if on_progress:
                on_progress(i + 1, total_chunks, f"Fetching chunk {i + 1}/{total_chunks}")

            try:
                raw_data = self.get_tx_data(tx_hash)

                # Parse chunk
                chunk_memory_id, chunk_index, chunk_bytes = parse_chunk_data(raw_data)

                if memory_id is None:
                    memory_id = chunk_memory_id
                elif memory_id != chunk_memory_id:
                    logger.warning(f"Memory ID mismatch at chunk {i}")

                # Store with index for ordering
                chunks_data.append((chunk_index, chunk_bytes))

                logger.debug(f"Chunk {i + 1}/{total_chunks}: {len(chunk_bytes)} bytes")

            except Exception as e:
                logger.error(f"Failed to fetch chunk {i}: {e}")
                return DownloadResult(
                    success=False,
                    memory_id=memory_id or "unknown",
                    mp4_path=None,
                    index_path=None,
                    total_chunks=total_chunks,
                    total_size=0,
                    error=str(e)
                )

            await asyncio.sleep(0.05)  # Rate limiting

        # Sort by index and combine
        chunks_data.sort(key=lambda x: x[0])
        combined_data = b''.join(chunk[1] for chunk in chunks_data)

        logger.info(f"Total downloaded: {len(combined_data):,} bytes")

        # Parse combined data
        # Format: [4 bytes: index_len][index_json][mp4_data]
        index_len = int.from_bytes(combined_data[:4], byteorder='big')
        index_bytes = combined_data[4:4 + index_len]
        mp4_data = combined_data[4 + index_len:]

        # Parse index
        try:
            index_data = json.loads(index_bytes.decode('utf-8'))
        except:
            logger.error("Failed to parse index JSON")
            return DownloadResult(
                success=False,
                memory_id=memory_id or "unknown",
                mp4_path=None,
                index_path=None,
                total_chunks=total_chunks,
                total_size=len(combined_data),
                error="Failed to parse index JSON"
            )

        # Save files
        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{memory_id}.mp4")
        index_path = os.path.join(output_dir, f"{memory_id}_index.json")

        with open(mp4_path, 'wb') as f:
            f.write(mp4_data)

        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        if on_progress:
            on_progress(total_chunks, total_chunks, "Download complete!")

        logger.info(f"Reconstructed: {mp4_path}")

        return DownloadResult(
            success=True,
            memory_id=memory_id,
            mp4_path=mp4_path,
            index_path=index_path,
            total_chunks=total_chunks,
            total_size=len(combined_data)
        )

    async def find_memories_by_wallet(self, wallet_address: str) -> List[Dict]:
        """
        Find all memory uploads from a specific wallet.

        Note: This requires indexing or event logs. For MVP, we'll scan
        recent transactions. In production, use The Graph or similar.
        """
        logger.info(f"Scanning for memories from {wallet_address}...")

        # This is a simplified implementation
        # In production, you'd use The Graph or event indexing
        memories = []

        try:
            # Get recent transactions (last 1000 blocks)
            latest_block = self.w3.eth.block_number
            start_block = max(0, latest_block - 1000)

            # This is expensive - production should use indexed data
            for block_num in range(start_block, latest_block + 1):
                block = self.w3.eth.get_block(block_num, full_transactions=True)

                for tx in block['transactions']:
                    if tx['from'].lower() == wallet_address.lower():
                        # Check if it's a memory transaction
                        if tx['input'][:8] == MEMORY_SIGNATURE.hex():
                            memory_id = tx['input'][8:40].decode('utf-8').rstrip('\x00')

                            # Add if not already found
                            existing = next((m for m in memories if m['memory_id'] == memory_id), None)
                            if existing:
                                existing['tx_hashes'].append(tx['hash'].hex())
                            else:
                                memories.append({
                                    'memory_id': memory_id,
                                    'tx_hashes': [tx['hash'].hex()],
                                    'wallet': wallet_address,
                                    'block': block_num
                                })

        except Exception as e:
            logger.error(f"Error scanning blockchain: {e}")

        return memories
