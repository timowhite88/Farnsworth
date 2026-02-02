"""
Example: Chain Memory Startup Integration

This shows how to integrate chain memory loading into your bot's startup.
Add this to your bot's main.py or entry point.
"""

import os
import asyncio
import logging

# Enable chain memory at startup
os.environ["CHAIN_MEMORY_ENABLED"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("example")


def start_bot_with_chain_memory():
    """
    Example startup sequence that integrates chain memory.

    This should be called before your bot's main initialization.
    """
    from farnsworth.integration.chain_memory import (
        prompt_memory_load,
        auto_load_memories
    )

    print("\n" + "=" * 60)
    print("  FARNSWORTH AI - Starting Up")
    print("=" * 60 + "\n")

    # Step 1: Prompt user for chain memory
    memories_to_load = prompt_memory_load(
        bot_type="farnsworth",
        auto_detect_wallet=True
    )

    # Step 2: Load memories if requested
    if memories_to_load:
        print("\nLoading chain memories...\n")

        def progress_callback(msg):
            print(f"  >> {msg}")

        success = asyncio.run(
            auto_load_memories(
                tx_ids_or_wallet=memories_to_load,
                bot_type="farnsworth",
                on_progress=progress_callback
            )
        )

        if success:
            print("\n  Chain memories loaded successfully!\n")
        else:
            print("\n  Warning: Some memories failed to load.\n")

    # Step 3: Continue with normal bot startup
    print("Continuing with normal startup...\n")

    # Your bot's initialization code here...
    # from farnsworth.core import FarnsworthBot
    # bot = FarnsworthBot()
    # bot.run()


def quick_push_example():
    """Quick example of pushing current memory to chain."""
    from farnsworth.integration.chain_memory import ChainMemory

    async def push():
        # Initialize with your wallet key
        cm = ChainMemory(
            wallet_key=os.getenv("MONAD_PRIVATE_KEY"),
            bot_type="farnsworth"
        )

        # Estimate cost first
        print("Estimating cost...")
        estimate = cm.estimate_push_cost()
        print(f"  Chunks: {estimate['num_chunks']}")
        print(f"  Cost: {estimate['total_cost_mon']:.4f} MON")
        print(f"  Cost: ${estimate['total_cost_usd']:.2f} USD")

        # Ask for confirmation
        confirm = input("\nProceed with upload? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

        # Push to chain
        print("\nPushing to chain...")

        def progress(current, total, status):
            print(f"  [{current}/{total}] {status}")

        record = await cm.push_memory(
            title="My Farnsworth Memory Backup",
            on_progress=progress
        )

        print("\n" + "=" * 60)
        print("  SUCCESS!")
        print("=" * 60)
        print(f"\n  Memory ID: {record.memory_id}")
        print(f"  Total Size: {record.total_size:,} bytes")
        print(f"  TX Count: {len(record.tx_hashes)}")
        print(f"\n  Save these TX hashes to recover your memory later:")
        for i, tx in enumerate(record.tx_hashes[:5]):
            print(f"    {i+1}. {tx}")
        if len(record.tx_hashes) > 5:
            print(f"    ... and {len(record.tx_hashes) - 5} more")

        # Export for backup
        export_file = f"memory_backup_{record.memory_id}.json"
        export_str = cm.export_tx_list(record.memory_id)
        with open(export_file, 'w') as f:
            f.write(export_str)
        print(f"\n  Exported backup to: {export_file}")

    asyncio.run(push())


def quick_pull_example():
    """Quick example of pulling memory from chain."""
    from farnsworth.integration.chain_memory import ChainMemory

    async def pull():
        cm = ChainMemory(bot_type="farnsworth")

        # Option 1: Pull from specific TX hashes
        tx_hashes = [
            "0x...",  # Replace with real TX hashes
            "0x...",
        ]

        # Option 2: Pull from memory ID in local records
        # memory_id = "abc123..."

        # Option 3: Pull all from wallet
        # wallet = "0x..."

        print("Pulling memory from chain...")

        def progress(current, total, status):
            print(f"  [{current}/{total}] {status}")

        package = await cm.pull_memory(
            tx_ids=tx_hashes,
            on_progress=progress
        )

        print(f"\nReconstructed memory package:")
        print(f"  Bot: {package.bot_name}")
        print(f"  Chunks: {len(package.chunks)}")
        print(f"  Has personality: {package.personality is not None}")

        # Load into Farnsworth
        confirm = input("\nLoad into Farnsworth memory? (y/n): ")
        if confirm.lower() == 'y':
            cm.load_into_farnsworth(package, merge=True)
            print("Memory loaded!")

    asyncio.run(pull())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "push":
            quick_push_example()
        elif command == "pull":
            quick_pull_example()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python startup_integration.py [push|pull]")
    else:
        start_bot_with_chain_memory()
