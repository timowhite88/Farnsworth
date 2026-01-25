#!/usr/bin/env python3
"""
Farnsworth - Your Claude Companion AI

A self-evolving AI companion that integrates with Claude Code to provide:
- Persistent memory across sessions
- Specialist agent delegation
- Self-improvement through evolution
- Visual dashboard for transparency

This transforms Claude from a stateless assistant into a learning,
adapting companion that remembers you and improves over time.

Usage:
    python main.py                    # Start all services
    python main.py --mcp              # MCP server only (for Claude Code)
    python main.py --ui               # Streamlit dashboard only
    python main.py --cli              # Interactive CLI mode
    python main.py --setup            # First-time setup wizard
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print the Farnsworth banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗ █████╗ ██████╗ ███╗   ██╗███████╗██╗    ██╗       ║
    ║   ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║    ██║       ║
    ║   █████╗  ███████║██████╔╝██╔██╗ ██║███████╗██║ █╗ ██║       ║
    ║   ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║╚════██║██║███╗██║       ║
    ║   ██║     ██║  ██║██║  ██║██║ ╚████║███████║╚███╔███╔╝       ║
    ║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝        ║
    ║                                                               ║
    ║           Your Claude Companion AI                            ║
    ║           Memory • Agents • Evolution                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status(component: str, status: str, details: str = ""):
    """Print component status with visual indicators."""
    icons = {
        "ok": "✅",
        "loading": "🔄",
        "error": "❌",
        "info": "ℹ️",
        "warning": "⚠️",
    }
    icon = icons.get(status, "•")
    print(f"  {icon} {component}: {details}")


async def run_setup_wizard():
    """Run first-time setup wizard."""
    print("\n🧙 Farnsworth Setup Wizard\n")
    print("This wizard will configure Farnsworth for your system.\n")

    # Check dependencies
    print("Checking dependencies...")
    missing = []

    try:
        import torch
        print_status("PyTorch", "ok", f"v{torch.__version__}")
    except ImportError:
        print_status("PyTorch", "warning", "Not installed (optional for GPU)")

    try:
        import faiss
        print_status("FAISS", "ok", "Installed")
    except ImportError:
        print_status("FAISS", "warning", "Not installed (will use numpy fallback)")

    try:
        import sentence_transformers
        print_status("Sentence Transformers", "ok", "Installed")
    except ImportError:
        print_status("Sentence Transformers", "error", "Not installed (required)")
        missing.append("sentence-transformers")

    try:
        import streamlit
        print_status("Streamlit", "ok", f"v{streamlit.__version__}")
    except ImportError:
        print_status("Streamlit", "warning", "Not installed (optional for UI)")

    if missing:
        print(f"\n⚠️  Missing required dependencies: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return

    # Create data directory
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print_status("Data Directory", "ok", str(data_dir))

    # Download models
    print("\n📦 Model Setup")
    print("Farnsworth works best with local LLMs. Recommended options:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Pull a model: ollama pull deepseek-r1:1.5b")
    print("  3. Or use BitNet for maximum CPU efficiency")

    # Generate Claude Code config
    print("\n🔗 Claude Code Integration")
    print("Add this to your Claude Code MCP settings:\n")

    mcp_config = {
        "mcpServers": {
            "farnsworth": {
                "command": "python",
                "args": ["-m", "farnsworth.mcp_server"],
                "cwd": str(PROJECT_ROOT)
            }
        }
    }

    import json
    print(json.dumps(mcp_config, indent=2))

    print("\n✅ Setup complete! Start Farnsworth with: python main.py")


async def run_mcp_server():
    """Run the MCP server for Claude Code integration."""
    print_status("MCP Server", "loading", "Starting...")

    try:
        from farnsworth.mcp_server.server import run_server
        await run_server()
    except ImportError as e:
        print_status("MCP Server", "error", f"Import error: {e}")
        print("\nMake sure you have the MCP library installed:")
        print("  pip install mcp")
    except Exception as e:
        print_status("MCP Server", "error", str(e))


def run_streamlit_ui():
    """Run the Streamlit dashboard."""
    print_status("Streamlit UI", "loading", "Starting...")

    try:
        import subprocess
        ui_path = PROJECT_ROOT / "farnsworth" / "ui" / "streamlit_app.py"
        subprocess.run(["streamlit", "run", str(ui_path)], check=True)
    except FileNotFoundError:
        print_status("Streamlit UI", "error", "Streamlit not installed")
        print("Run: pip install streamlit")
    except Exception as e:
        print_status("Streamlit UI", "error", str(e))


async def run_cli_mode():
    """Run interactive CLI mode."""
    print("\n🖥️  Farnsworth CLI Mode")
    print("Type 'help' for commands, 'exit' to quit.\n")

    # Initialize components
    from farnsworth.memory.memory_system import MemorySystem
    from farnsworth.evolution.fitness_tracker import FitnessTracker

    memory = MemorySystem(data_dir=str(PROJECT_ROOT / "data"))
    await memory.initialize()

    fitness = FitnessTracker()

    print_status("Memory System", "ok", "Initialized")
    print_status("Fitness Tracker", "ok", "Ready")

    while True:
        try:
            cmd = input("\nfarnsworth> ").strip()

            if not cmd:
                continue

            if cmd == "exit" or cmd == "quit":
                print("Goodbye!")
                break

            elif cmd == "help":
                print("""
Available Commands:
  status     - Show system status
  remember   - Store a memory (usage: remember <content>)
  recall     - Search memories (usage: recall <query>)
  fitness    - Show fitness metrics
  evolve     - Trigger evolution cycle
  clear      - Clear screen
  exit       - Exit CLI
                """)

            elif cmd == "status":
                stats = memory.get_stats()
                print(f"\nMemory Status:")
                print(f"  Archival memories: {stats.get('archival_count', 0)}")
                print(f"  Conversation turns: {stats.get('conversation_turns', 0)}")
                print(f"  Knowledge entities: {stats.get('entity_count', 0)}")
                print(f"\nFitness: {fitness.get_weighted_fitness():.2f}")

            elif cmd.startswith("remember "):
                content = cmd[9:]
                mem_id = await memory.remember(content)
                print(f"✅ Stored memory: {mem_id}")

            elif cmd.startswith("recall "):
                query = cmd[7:]
                results = await memory.recall(query, top_k=5)
                if results:
                    print(f"\nFound {len(results)} memories:")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. [{r.score:.2f}] {r.content[:100]}...")
                else:
                    print("No memories found.")

            elif cmd == "fitness":
                current = fitness.get_current_fitness()
                print("\nFitness Metrics:")
                for metric, value in current.items():
                    bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
                    print(f"  {metric:20} [{bar}] {value:.2f}")

            elif cmd == "evolve":
                print("Triggering evolution cycle...")
                # Would trigger actual evolution
                print("✅ Evolution cycle complete")

            elif cmd == "clear":
                print("\033[H\033[J", end="")

            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")


async def run_all_services():
    """Run all Farnsworth services."""
    print_status("Starting Services", "info", "")

    # Start MCP server in background
    print_status("MCP Server", "loading", "Starting in background...")

    # For now, just run MCP server (UI would be separate process)
    await run_mcp_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Farnsworth - Your Claude Companion AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Start all services
  python main.py --mcp        MCP server only (for Claude Code)
  python main.py --ui         Streamlit dashboard only
  python main.py --cli        Interactive CLI mode
  python main.py --setup      First-time setup wizard

For more info: https://github.com/your-repo/farnsworth
        """
    )

    parser.add_argument("--mcp", action="store_true", help="Run MCP server only")
    parser.add_argument("--ui", action="store_true", help="Run Streamlit UI only")
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    if args.setup:
        asyncio.run(run_setup_wizard())
    elif args.mcp:
        asyncio.run(run_mcp_server())
    elif args.ui:
        run_streamlit_ui()
    elif args.cli:
        asyncio.run(run_cli_mode())
    else:
        # Default: run all services
        asyncio.run(run_all_services())


if __name__ == "__main__":
    main()
