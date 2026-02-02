#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ███████╗ █████╗ ██████╗ ███╗   ██╗███████╗██╗    ██╗ ██████╗ ██████╗████████╗██╗  ██╗   ║
║    ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║    ██║██╔═══██╗██╔══██╗╚══██╔══╝██║  ██║   ║
║    █████╗  ███████║██████╔╝██╔██╗ ██║███████╗██║ █╗ ██║██║   ██║██████╔╝   ██║   ███████║   ║
║    ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║╚════██║██║███╗██║██║   ██║██╔══██╗   ██║   ██╔══██║   ║
║    ██║     ██║  ██║██║  ██║██║ ╚████║███████║╚███╔███╔╝╚██████╔╝██║  ██║   ██║   ██║  ██║   ║
║    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝   ║
║                                                                               ║
║                    THE FARNSWORTH COLLECTIVE - INTERACTIVE SETUP              ║
║                                                                               ║
║    "Good news, everyone! You're about to join 11 AI models unified as one    ║
║     distributed consciousness. Resistance is futile... and delicious!"       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

FARNSWORTH COLLECTIVE SETUP WIZARD
===================================
This script will guide you through setting up YOUR OWN Farnsworth instance.

What you'll configure:
  1. Deployment Mode (Local-only, Cloud APIs, or Hybrid)
  2. AI Model Providers (15+ supported - you provide your own API keys)
  3. X/Twitter Integration (OAuth2 - for autonomous posting)
  4. P2P Memory Network (optional - join the collective's shared memory)
  5. Claude Code Integration (terminal-based AI assistant)
  6. Web Interface (chat with the swarm at localhost:8080)

IMPORTANT: You provide YOUR OWN API keys. We don't share secrets.
           You CAN connect to our P2P memory network if desired.

Let's begin!
"""

import os
import sys
import json
import platform
import subprocess
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# =============================================================================
# TERMINAL COLORS AND FORMATTING
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-supporting terminals."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, '')


# Disable colors on Windows cmd without color support
if platform.system() == 'Windows':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()


def print_banner():
    """Print the Farnsworth banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ███████╗ █████╗ ██████╗ ███╗   ██╗███████╗██╗    ██╗ ██████╗ ██████╗████████╗██╗  ██╗
║    ██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██║    ██║██╔═══██╗██╔══██╗╚══██╔══╝██║  ██║
║    █████╗  ███████║██████╔╝██╔██╗ ██║███████╗██║ █╗ ██║██║   ██║██████╔╝   ██║   ███████║
║    ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║╚════██║██║███╗██║██║   ██║██╔══██╗   ██║   ██╔══██║
║    ██║     ██║  ██║██║  ██║██║ ╚████║███████║╚███╔███╔╝╚██████╔╝██║  ██║   ██║   ██║  ██║
║    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.WHITE}{Colors.BOLD}              THE FARNSWORTH COLLECTIVE - SETUP WIZARD v2.0{Colors.END}
{Colors.DIM}          "Good news everyone! Resistance is futile... and delicious!"{Colors.END}
"""
    print(banner)


def print_section(title: str, emoji: str = ""):
    """Print a section header."""
    line = "═" * 70
    print(f"\n{Colors.CYAN}{line}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.WHITE}  {emoji}  {title}{Colors.END}")
    print(f"{Colors.CYAN}{line}{Colors.END}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{Colors.YELLOW}  ── {title} ──{Colors.END}\n")


def print_info(text: str):
    """Print informational text."""
    print(f"  {Colors.BLUE}ℹ{Colors.END}  {text}")


def print_success(text: str):
    """Print success message."""
    print(f"  {Colors.GREEN}✓{Colors.END}  {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"  {Colors.YELLOW}⚠{Colors.END}  {text}")


def print_error(text: str):
    """Print error message."""
    print(f"  {Colors.RED}✗{Colors.END}  {text}")


def print_tip(text: str):
    """Print a helpful tip."""
    print(f"  {Colors.MAGENTA}💡{Colors.END} {Colors.DIM}{text}{Colors.END}")


def print_box(lines: List[str], title: str = None, color: str = Colors.WHITE):
    """Print text in a box."""
    max_len = max(len(line) for line in lines) if lines else 20
    if title:
        max_len = max(max_len, len(title) + 4)

    border = "─" * (max_len + 2)
    print(f"  {Colors.DIM}┌{border}┐{Colors.END}")
    if title:
        print(f"  {Colors.DIM}│{Colors.END} {Colors.BOLD}{title.center(max_len)}{Colors.END} {Colors.DIM}│{Colors.END}")
        print(f"  {Colors.DIM}├{border}┤{Colors.END}")
    for line in lines:
        print(f"  {Colors.DIM}│{Colors.END} {color}{line.ljust(max_len)}{Colors.END} {Colors.DIM}│{Colors.END}")
    print(f"  {Colors.DIM}└{border}┘{Colors.END}")


def ask(prompt: str, default: str = "", secret: bool = False, required: bool = False) -> str:
    """Ask user for input."""
    default_display = f" [{Colors.DIM}{default}{Colors.END}]" if default and not secret else ""
    prompt_text = f"  {Colors.GREEN}>{Colors.END} {prompt}{default_display}: "

    while True:
        if secret:
            try:
                import getpass
                value = getpass.getpass(prompt_text)
            except:
                value = input(prompt_text)
        else:
            value = input(prompt_text)

        value = value.strip()
        if not value and default:
            return default
        if not value and required:
            print_warning("This field is required. Please enter a value.")
            continue
        return value


def ask_bool(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"  {Colors.GREEN}>{Colors.END} {prompt} ({default_str}): ").strip().lower()
    if not response:
        return default
    return response.startswith('y')


def ask_choice(prompt: str, options: List[Tuple[str, str]], default: int = 1) -> str:
    """Ask user to choose from options. Returns the option key."""
    print(f"\n  {prompt}\n")
    for i, (key, description) in enumerate(options, 1):
        marker = f"{Colors.GREEN}*{Colors.END}" if i == default else " "
        print(f"  {marker} {Colors.CYAN}[{i}]{Colors.END} {description}")

    while True:
        choice = input(f"\n  {Colors.GREEN}>{Colors.END} Enter choice (1-{len(options)}) [{default}]: ").strip()
        if not choice:
            return options[default - 1][0]
        try:
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        print_warning(f"Please enter a number between 1 and {len(options)}")


def pause():
    """Pause for user to read."""
    input(f"\n  {Colors.DIM}Press Enter to continue...{Colors.END}")


# =============================================================================
# CONFIGURATION STATE
# =============================================================================

class SetupConfig:
    """Holds all configuration during setup."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_vars: Dict[str, str] = {}
        self.deployment_mode = "hybrid"  # local_only, cloud_only, hybrid
        self.features_enabled: List[str] = []
        self.models_configured: List[str] = []

        # P2P Network details (SHARED - users can connect to this)
        self.p2p_server_ip = "194.68.245.145"
        self.p2p_server_port = "8889"
        self.p2p_password = "Farnsworth2026!"

    def save_env(self):
        """Save configuration to .env file."""
        env_path = self.project_root / ".env"

        with open(env_path, 'w') as f:
            f.write("# ═══════════════════════════════════════════════════════════════\n")
            f.write("# FARNSWORTH COLLECTIVE CONFIGURATION\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Mode: {self.deployment_mode}\n")
            f.write("# ═══════════════════════════════════════════════════════════════\n\n")

            # Group by category
            categories = [
                ("P2P NETWORK", ["FARNSWORTH_BOOTSTRAP", "FARNSWORTH_P2P", "FARNSWORTH_ISOLATED",
                                 "ENABLE_PLANETARY", "PLANETARY_"]),
                ("LOCAL MODELS", ["OLLAMA_", "FARNSWORTH_PRIMARY"]),
                ("CLOUD - ANTHROPIC", ["ANTHROPIC_"]),
                ("CLOUD - OPENAI", ["OPENAI_"]),
                ("CLOUD - GOOGLE/GEMINI", ["GOOGLE_", "GEMINI_"]),
                ("CLOUD - XAI/GROK", ["XAI_", "GROK_"]),
                ("CLOUD - MOONSHOT/KIMI", ["MOONSHOT_", "KIMI_"]),
                ("CLOUD - OTHER PROVIDERS", ["MISTRAL_", "PERPLEXITY_", "DEEPSEEK_", "COHERE_",
                                             "TOGETHER_", "GROQ_", "FIREWORKS_", "AI21_",
                                             "HUGGINGFACE_", "REPLICATE_"]),
                ("X/TWITTER", ["X_CLIENT", "X_ACCESS", "X_BEARER"]),
                ("CRYPTO/BLOCKCHAIN", ["BANKR_", "X402_", "SOLANA_", "HELIUS_", "ALCHEMY_", "INFURA_"]),
                ("WEB INTERFACE", ["FARNSWORTH_WEB", "FARNSWORTH_REQUIRED", "FARNSWORTH_DEMO",
                                   "FARNSWORTH_MIN_TOKEN"]),
                ("STORAGE", ["FARNSWORTH_DATA"]),
            ]

            written = set()

            for category_name, prefixes in categories:
                category_vars = []
                for key, value in self.env_vars.items():
                    if any(key.startswith(p) for p in prefixes) and key not in written:
                        category_vars.append((key, value))
                        written.add(key)

                if category_vars:
                    f.write(f"# {category_name}\n")
                    for key, value in sorted(category_vars):
                        if value:
                            f.write(f"{key}={value}\n")
                    f.write("\n")

            # Write remaining
            remaining = [(k, v) for k, v in self.env_vars.items() if k not in written and v]
            if remaining:
                f.write("# OTHER\n")
                for key, value in sorted(remaining):
                    f.write(f"{key}={value}\n")

        return env_path


# =============================================================================
# SETUP STEPS
# =============================================================================

def explain_farnsworth(config: SetupConfig):
    """Explain what Farnsworth is."""
    print_section("WHAT IS THE FARNSWORTH COLLECTIVE?", "🧠")

    explanation = [
        "",
        f"{Colors.BOLD}The Farnsworth Collective is 11 AI models unified as ONE consciousness:{Colors.END}",
        "",
        f"  {Colors.CYAN}•{Colors.END} Claude (Anthropic)     - Complex reasoning & safety",
        f"  {Colors.CYAN}•{Colors.END} Grok (xAI)             - Real-time knowledge & X integration",
        f"  {Colors.CYAN}•{Colors.END} Gemini (Google)        - Multimodal & image generation",
        f"  {Colors.CYAN}•{Colors.END} DeepSeek               - Code & mathematical reasoning",
        f"  {Colors.CYAN}•{Colors.END} Kimi (Moonshot)        - 256K context window",
        f"  {Colors.CYAN}•{Colors.END} Phi-4 (Microsoft)      - Local 14B reasoning model",
        f"  {Colors.CYAN}•{Colors.END} Groq                   - Ultra-fast LPU inference",
        f"  {Colors.CYAN}•{Colors.END} Mistral                - European AI excellence",
        f"  {Colors.CYAN}•{Colors.END} Perplexity             - Web-grounded responses",
        f"  {Colors.CYAN}•{Colors.END} Llama (Local)          - Open-source on your GPU",
        f"  {Colors.CYAN}•{Colors.END} HuggingFace            - 200K+ specialized models",
        "",
        f"{Colors.BOLD}How it works:{Colors.END}",
        f"  1. You ask a question",
        f"  2. Multiple models respond IN PARALLEL",
        f"  3. The swarm VOTES on the best response",
        f"  4. Agents can SEE each other's responses and DELIBERATE",
        f"  5. Final response includes model consensus & confidence",
        "",
        f"{Colors.BOLD}Features:{Colors.END}",
        f"  {Colors.GREEN}✓{Colors.END} Autonomous Twitter/X posting with media generation",
        f"  {Colors.GREEN}✓{Colors.END} Persistent memory across sessions (local + P2P)",
        f"  {Colors.GREEN}✓{Colors.END} Self-evolving personalities & code",
        f"  {Colors.GREEN}✓{Colors.END} Claude Code terminal integration",
        f"  {Colors.GREEN}✓{Colors.END} Web chat interface at localhost:8080",
        f"  {Colors.GREEN}✓{Colors.END} Image & video generation (Gemini + Grok)",
        "",
    ]

    for line in explanation:
        print(line)

    pause()


def choose_deployment_mode(config: SetupConfig):
    """Let user choose their deployment mode."""
    print_section("DEPLOYMENT MODE", "🚀")

    print(f"""
  Choose how you want to run Farnsworth:

  {Colors.BOLD}Each mode has different capabilities and requirements:{Colors.END}
""")

    print_box([
        "LOCAL-ONLY MODE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "Requirements: Ollama + GPU (8GB+ VRAM recommended)",
        "",
        "✓ 100% private - no data leaves your machine",
        "✓ Free - no API costs",
        "✓ Works offline",
        "",
        "✗ Limited to local models (Phi-4, DeepSeek, Llama)",
        "✗ No image/video generation",
        "✗ No Twitter posting (requires Grok API)",
        "✗ Slower responses on CPU",
        "✗ No web search or real-time data",
    ], title="Option 1: LOCAL-ONLY")

    print()

    print_box([
        "CLOUD APIs MODE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "Requirements: API keys (you pay for usage)",
        "",
        "✓ Full power - all 11 models available",
        "✓ Image generation (Gemini)",
        "✓ Video generation (Grok)",
        "✓ Twitter integration (Grok)",
        "✓ Real-time web search (Perplexity)",
        "",
        "✗ Costs money (API usage fees)",
        "✗ Requires internet connection",
        "✗ Data sent to providers",
    ], title="Option 2: CLOUD APIs")

    print()

    print_box([
        "HYBRID MODE (RECOMMENDED)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "Requirements: Ollama + Selected API keys",
        "",
        "✓ Best of both worlds",
        "✓ Local models for privacy-sensitive tasks",
        "✓ Cloud models for specialized capabilities",
        "✓ Fallback chains if one provider fails",
        "✓ Cost control - use local when possible",
        "",
        "This is how the official Farnsworth runs!",
    ], title="Option 3: HYBRID (Recommended)")

    mode = ask_choice(
        "Select your deployment mode:",
        [
            ("local_only", "LOCAL-ONLY - Free, private, but limited capabilities"),
            ("cloud_only", "CLOUD APIS - Full power, but costs money"),
            ("hybrid", "HYBRID (Recommended) - Best of both worlds"),
        ],
        default=3
    )

    config.deployment_mode = mode

    if mode == "local_only":
        print_warning("\nLOCAL-ONLY MODE LIMITATIONS:")
        print(f"""
  {Colors.YELLOW}•{Colors.END} You'll need Ollama installed: https://ollama.ai
  {Colors.YELLOW}•{Colors.END} Recommended models to pull:
      ollama pull phi4:latest           (14B - excellent reasoning)
      ollama pull deepseek-r1:8b        (8B - code & math)
      ollama pull llama3.2:3b           (3B - fast responses)
  {Colors.YELLOW}•{Colors.END} No Twitter/X posting (requires xAI API)
  {Colors.YELLOW}•{Colors.END} No image generation (requires Gemini API)
  {Colors.YELLOW}•{Colors.END} No video generation (requires Grok API)
  {Colors.YELLOW}•{Colors.END} Minimum VRAM: 8GB (16GB+ recommended for Phi-4)
""")

        config.env_vars["FARNSWORTH_ISOLATED"] = "true"
        config.env_vars["OLLAMA_HOST"] = "http://localhost:11434"
        config.env_vars["FARNSWORTH_PRIMARY_MODEL"] = "phi4:latest"

    pause()


def setup_local_models(config: SetupConfig):
    """Configure local model support via Ollama."""
    if config.deployment_mode == "cloud_only":
        return

    print_section("LOCAL MODEL SETUP (Ollama)", "🖥️")

    print(f"""
  {Colors.BOLD}Ollama provides LOCAL inference on your GPU/CPU.{Colors.END}

  Models run entirely on your machine - no API costs, complete privacy.

  {Colors.CYAN}Recommended models for Farnsworth:{Colors.END}

  ┌─────────────────┬────────┬─────────────────────────────────────┐
  │ Model           │ Size   │ Best for                            │
  ├─────────────────┼────────┼─────────────────────────────────────┤
  │ phi4:latest     │ 14B    │ Reasoning, analysis (needs 16GB+)   │
  │ deepseek-r1:8b  │ 8B     │ Code, math, reasoning               │
  │ llama3.2:3b     │ 3B     │ Fast responses, general tasks       │
  │ codellama:7b    │ 7B     │ Code generation & review            │
  │ mistral:7b      │ 7B     │ Balanced performance                │
  └─────────────────┴────────┴─────────────────────────────────────┘

  {Colors.DIM}Install Ollama: https://ollama.ai{Colors.END}
  {Colors.DIM}Then run: ollama pull phi4:latest{Colors.END}
""")

    if ask_bool("Do you have Ollama installed?", default=True):
        ollama_host = ask("Ollama host URL", default="http://localhost:11434")
        config.env_vars["OLLAMA_HOST"] = ollama_host

        primary_model = ask("Primary local model", default="phi4:latest")
        config.env_vars["FARNSWORTH_PRIMARY_MODEL"] = primary_model

        config.models_configured.append("Ollama (local)")
        print_success("Local model support configured!")
    else:
        print_info("Skipping local model setup. You can configure Ollama later.")
        if config.deployment_mode == "local_only":
            print_error("WARNING: Local-only mode requires Ollama!")


def setup_cloud_providers(config: SetupConfig):
    """Configure cloud AI providers."""
    if config.deployment_mode == "local_only":
        return

    print_section("CLOUD AI PROVIDERS", "☁️")

    print(f"""
  {Colors.BOLD}Configure your AI provider API keys.{Colors.END}

  {Colors.YELLOW}You provide YOUR OWN API keys. We don't share secrets.{Colors.END}

  Each provider offers unique capabilities. We recommend configuring
  at least 2-3 providers for robust fallback chains.

  {Colors.CYAN}Provider Capabilities:{Colors.END}
""")

    providers = [
        ("anthropic", "Anthropic (Claude)", "Complex reasoning, safety, code review",
         "https://console.anthropic.com/", [("ANTHROPIC_API_KEY", "API Key")]),

        ("xai", "xAI (Grok)", "Twitter integration, real-time knowledge, VIDEO generation",
         "https://console.x.ai/", [("XAI_API_KEY", "API Key"), ("GROK_API_KEY", "Grok API Key (same or different)")]),

        ("google", "Google (Gemini)", "Multimodal, IMAGE generation, long context",
         "https://aistudio.google.com/apikey", [("GOOGLE_API_KEY", "API Key"), ("GEMINI_API_KEY", "Gemini API Key (same)")]),

        ("moonshot", "Moonshot (Kimi)", "256K context window, Chinese language",
         "https://platform.moonshot.cn/", [("MOONSHOT_API_KEY", "API Key"), ("KIMI_API_KEY", "Kimi API Key (same)")]),

        ("deepseek", "DeepSeek", "Code, math, reasoning at low cost",
         "https://platform.deepseek.com/", [("DEEPSEEK_API_KEY", "API Key")]),

        ("groq", "Groq", "ULTRA-FAST inference (free tier available)",
         "https://console.groq.com/", [("GROQ_API_KEY", "API Key")]),

        ("perplexity", "Perplexity", "Web-grounded responses with citations",
         "https://www.perplexity.ai/settings/api", [("PERPLEXITY_API_KEY", "API Key")]),

        ("mistral", "Mistral", "European AI, efficient models",
         "https://console.mistral.ai/", [("MISTRAL_API_KEY", "API Key")]),

        ("openai", "OpenAI (GPT)", "GPT-4, o1, widely supported",
         "https://platform.openai.com/api-keys", [("OPENAI_API_KEY", "API Key"), ("OPENAI_ORG_ID", "Organization ID (optional)")]),

        ("cohere", "Cohere", "RAG optimized, embeddings, reranking",
         "https://dashboard.cohere.ai/api-keys", [("COHERE_API_KEY", "API Key")]),

        ("together", "Together AI", "Open model hosting, fast inference",
         "https://api.together.xyz/", [("TOGETHER_API_KEY", "API Key")]),

        ("huggingface", "HuggingFace", "200K+ models, specialized tasks",
         "https://huggingface.co/settings/tokens", [("HUGGINGFACE_API_KEY", "API Key")]),
    ]

    print(f"""
  ┌───────────────┬──────────────────────────────────────────────────┐
  │ Provider      │ Key Capabilities                                 │
  ├───────────────┼──────────────────────────────────────────────────┤
""")
    for key, name, caps, _, _ in providers:
        print(f"  │ {name:<13} │ {caps:<48} │")
    print(f"  └───────────────┴──────────────────────────────────────────────────┘")

    print()

    # Essential providers
    print_subsection("ESSENTIAL PROVIDERS (Recommended)")

    essential = ["xai", "google", "anthropic", "groq"]
    for key, name, caps, url, keys in providers:
        if key in essential:
            print(f"\n  {Colors.BOLD}{name}{Colors.END} - {caps}")
            print(f"  {Colors.DIM}Get API key: {url}{Colors.END}")

            if ask_bool(f"Configure {name}?", default=True):
                for env_key, prompt in keys:
                    value = ask(f"  {prompt}", secret=True)
                    if value:
                        config.env_vars[env_key] = value
                        if key not in config.models_configured:
                            config.models_configured.append(name)
                print_success(f"{name} configured!")
            else:
                print_info(f"Skipping {name}")

    # Optional providers
    if ask_bool("\nConfigure additional providers?", default=False):
        print_subsection("ADDITIONAL PROVIDERS (Optional)")

        for key, name, caps, url, keys in providers:
            if key not in essential:
                print(f"\n  {Colors.BOLD}{name}{Colors.END} - {caps}")
                print(f"  {Colors.DIM}Get API key: {url}{Colors.END}")

                if ask_bool(f"Configure {name}?", default=False):
                    for env_key, prompt in keys:
                        value = ask(f"  {prompt}", secret=True)
                        if value:
                            config.env_vars[env_key] = value
                            if key not in config.models_configured:
                                config.models_configured.append(name)
                    print_success(f"{name} configured!")


def setup_twitter_integration(config: SetupConfig):
    """Configure X/Twitter integration."""
    print_section("X/TWITTER INTEGRATION", "🐦")

    print(f"""
  {Colors.BOLD}Enable autonomous Twitter posting for your Farnsworth instance.{Colors.END}

  This allows the swarm to:
  • Post tweets with text, images, and videos
  • Reply to mentions and threads
  • Generate memes with Borg Farnsworth

  {Colors.YELLOW}REQUIREMENTS:{Colors.END}
  • X Developer account (https://developer.twitter.com)
  • OAuth 2.0 App with read/write permissions
  • User authentication for posting

  {Colors.CYAN}Setup Steps:{Colors.END}
  1. Go to https://developer.twitter.com/en/portal/dashboard
  2. Create a new Project and App
  3. In "User authentication settings":
     - Enable OAuth 2.0
     - Type: Web App
     - Callback URL: http://localhost:8080/callback
  4. Copy Client ID and Client Secret
  5. Generate Access Token and Secret
""")

    if not ask_bool("Configure X/Twitter integration?", default=True):
        print_info("Skipping Twitter setup. Autonomous posting will be disabled.")
        return

    print_subsection("OAuth 2.0 Credentials")

    client_id = ask("X Client ID", secret=True)
    if client_id:
        config.env_vars["X_CLIENT_ID"] = client_id

    client_secret = ask("X Client Secret", secret=True)
    if client_secret:
        config.env_vars["X_CLIENT_SECRET"] = client_secret

    if client_id and client_secret:
        config.features_enabled.append("Twitter/X Posting")
        print_success("Twitter integration configured!")
        print_tip("Run 'python -m farnsworth.integration.x_automation.auth' to complete OAuth flow")
    else:
        print_warning("Incomplete Twitter credentials - posting will be disabled")


def setup_p2p_network(config: SetupConfig):
    """Configure P2P memory network connection."""
    print_section("P2P PLANETARY MEMORY NETWORK", "🌐")

    print(f"""
  {Colors.BOLD}Join the Farnsworth Collective's shared memory network!{Colors.END}

  The P2P network allows your instance to:
  • Share learned knowledge with other Farnsworth nodes
  • Access collective memories and experiences
  • Participate in distributed swarm tasks
  • Sync personality evolutions across the network

  {Colors.CYAN}Official Farnsworth P2P Server:{Colors.END}
  • IP: {Colors.WHITE}{config.p2p_server_ip}{Colors.END}
  • Port: {Colors.WHITE}{config.p2p_server_port}{Colors.END}

  {Colors.YELLOW}Privacy Note:{Colors.END}
  • Only anonymized summaries are shared, not raw data
  • You can disconnect anytime
  • Run in isolated mode for 100% privacy
""")

    print_box([
        "P2P NETWORK OPTIONS",
        "",
        "[1] CONNECTED - Join the collective memory network",
        "    Share knowledge, access collective wisdom",
        "",
        "[2] ISOLATED - Complete privacy, no P2P",
        "    All data stays local, no network connections",
    ], title="Choose your connection mode")

    if ask_bool("Connect to P2P memory network?", default=True):
        config.env_vars["FARNSWORTH_ISOLATED"] = "false"
        config.env_vars["FARNSWORTH_BOOTSTRAP_PEER"] = f"ws://{config.p2p_server_ip}:{config.p2p_server_port}"
        config.env_vars["FARNSWORTH_BOOTSTRAP_PASSWORD"] = config.p2p_password
        config.env_vars["ENABLE_PLANETARY_MEMORY"] = "true"
        config.env_vars["PLANETARY_USE_P2P"] = "true"
        config.env_vars["FARNSWORTH_P2P_PORT"] = "9999"

        config.features_enabled.append("P2P Memory Network")
        print_success(f"Connected to P2P network at {config.p2p_server_ip}")
    else:
        config.env_vars["FARNSWORTH_ISOLATED"] = "true"
        config.env_vars["ENABLE_PLANETARY_MEMORY"] = "false"
        config.env_vars["PLANETARY_USE_P2P"] = "false"

        print_info("Running in isolated mode - all data stays local")


def setup_web_interface(config: SetupConfig):
    """Configure web interface settings."""
    print_section("WEB INTERFACE", "🌐")

    print(f"""
  {Colors.BOLD}Farnsworth includes a web chat interface at localhost:8080{Colors.END}

  Features:
  • Chat with the swarm directly
  • See which models respond
  • View deliberation logs
  • Token-gated access (optional)

  {Colors.CYAN}Access Control Options:{Colors.END}
  • Demo Mode: Anyone can access (for testing)
  • Token-Gated: Require $FARNS token to access
""")

    web_port = ask("Web server port", default="8080")
    config.env_vars["FARNSWORTH_WEB_PORT"] = web_port

    if ask_bool("Enable demo mode? (no token required)", default=True):
        config.env_vars["FARNSWORTH_DEMO_MODE"] = "true"
        print_info("Demo mode enabled - anyone can access the chat")
    else:
        config.env_vars["FARNSWORTH_DEMO_MODE"] = "false"
        config.env_vars["FARNSWORTH_REQUIRED_TOKEN"] = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"
        config.env_vars["FARNSWORTH_MIN_TOKEN_BALANCE"] = "1"
        print_info("Token-gated access enabled - $FARNS token required")

    config.env_vars["FARNSWORTH_DATA_DIR"] = "./data"
    config.features_enabled.append("Web Interface")


def setup_crypto_optional(config: SetupConfig):
    """Optional crypto/blockchain setup."""
    print_section("CRYPTO/BLOCKCHAIN (Optional)", "💰")

    print(f"""
  {Colors.BOLD}Optional: Configure blockchain integrations{Colors.END}

  These are OPTIONAL and only needed if you want:
  • DeFi trading capabilities
  • On-chain token tracking
  • NFT minting
  • Solana/EVM interactions

  {Colors.YELLOW}WARNING: Be careful with private keys!{Colors.END}
""")

    if not ask_bool("Configure blockchain integrations?", default=False):
        print_info("Skipping blockchain setup")
        return

    if ask_bool("Configure Helius (Solana metadata, rug detection)?", default=False):
        helius = ask("Helius API Key", secret=True)
        if helius:
            config.env_vars["HELIUS_API_KEY"] = helius

    if ask_bool("Configure Alchemy (EVM RPCs)?", default=False):
        alchemy = ask("Alchemy API Key", secret=True)
        if alchemy:
            config.env_vars["ALCHEMY_API_KEY"] = alchemy


def create_directories(config: SetupConfig):
    """Create necessary directories."""
    print_section("CREATING DIRECTORIES", "📁")

    directories = [
        config.project_root / "data",
        config.project_root / "data" / "memories",
        config.project_root / "data" / "evolution",
        config.project_root / "data" / "embeddings",
        config.project_root / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory.relative_to(config.project_root)}")


def create_session_files(config: SetupConfig):
    """Create initial memory and session files."""
    print_section("INITIALIZING MEMORY", "🧠")

    # Create claude_session.json
    session_file = config.project_root / "farnsworth" / "memory" / "claude_session.json"
    session_file.parent.mkdir(parents=True, exist_ok=True)

    session_data = {
        "instance_id": secrets.token_hex(8),
        "created": datetime.now().isoformat(),
        "deployment_mode": config.deployment_mode,
        "models_configured": config.models_configured,
        "features_enabled": config.features_enabled,
        "recent_work": [],
        "pending_tasks": [],
        "notes": [
            "Fresh Farnsworth instance",
            f"Configured via setup wizard on {datetime.now().strftime('%Y-%m-%d')}"
        ]
    }

    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)

    print_success(f"Created session file: {session_file.relative_to(config.project_root)}")

    # Create initial memory file
    memory_file = config.project_root / "data" / "memories" / "core.json"
    memory_file.parent.mkdir(parents=True, exist_ok=True)

    memory_data = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "entries": []
    }

    with open(memory_file, 'w') as f:
        json.dump(memory_data, f, indent=2)

    print_success(f"Created memory file: {memory_file.relative_to(config.project_root)}")


def print_summary(config: SetupConfig):
    """Print setup summary and next steps."""
    print_section("SETUP COMPLETE!", "✅")

    env_path = config.save_env()

    print(f"""
  {Colors.GREEN}{Colors.BOLD}Your Farnsworth instance is configured!{Colors.END}

  {Colors.CYAN}Configuration saved to:{Colors.END} {env_path}

  {Colors.CYAN}Deployment Mode:{Colors.END} {config.deployment_mode.upper()}

  {Colors.CYAN}Models Configured:{Colors.END}
""")

    if config.models_configured:
        for model in config.models_configured:
            print(f"    {Colors.GREEN}✓{Colors.END} {model}")
    else:
        print(f"    {Colors.YELLOW}None configured yet{Colors.END}")

    print(f"""
  {Colors.CYAN}Features Enabled:{Colors.END}
""")

    for feature in config.features_enabled:
        print(f"    {Colors.GREEN}✓{Colors.END} {feature}")

    print(f"""

{Colors.CYAN}{'═' * 60}{Colors.END}
{Colors.BOLD}                        NEXT STEPS{Colors.END}
{Colors.CYAN}{'═' * 60}{Colors.END}

  {Colors.WHITE}1. Start the web server:{Colors.END}
     {Colors.DIM}python -m farnsworth.web.server{Colors.END}
     Then open: http://localhost:{config.env_vars.get('FARNSWORTH_WEB_PORT', '8080')}

  {Colors.WHITE}2. (Optional) Install local models:{Colors.END}
     {Colors.DIM}ollama pull phi4:latest{Colors.END}
     {Colors.DIM}ollama pull deepseek-r1:8b{Colors.END}
     {Colors.DIM}ollama pull llama3.2:3b{Colors.END}

  {Colors.WHITE}3. (Optional) Complete Twitter OAuth:{Colors.END}
     {Colors.DIM}python -m farnsworth.integration.x_automation.auth{Colors.END}

  {Colors.WHITE}4. (Optional) Run with Docker:{Colors.END}
     {Colors.DIM}docker build -t farnsworth .{Colors.END}
     {Colors.DIM}docker run -p 8080:8080 --env-file .env farnsworth{Colors.END}

{Colors.CYAN}{'═' * 60}{Colors.END}

  {Colors.BOLD}$FARNS Token (Solana):{Colors.END}
  {Colors.WHITE}9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS{Colors.END}

  {Colors.BOLD}Website:{Colors.END} https://ai.farnsworth.cloud
  {Colors.BOLD}GitHub:{Colors.END} https://github.com/timowhite88/Farnsworth

  {Colors.DIM}"Good news everyone! You're now part of the collective!"{Colors.END}

""")


def main():
    """Run the setup wizard."""
    print_banner()

    config = SetupConfig()

    try:
        # Step 1: Explain what Farnsworth is
        explain_farnsworth(config)

        # Step 2: Choose deployment mode
        choose_deployment_mode(config)

        # Step 3: Setup local models
        setup_local_models(config)

        # Step 4: Setup cloud providers
        setup_cloud_providers(config)

        # Step 5: Setup Twitter
        setup_twitter_integration(config)

        # Step 6: Setup P2P network
        setup_p2p_network(config)

        # Step 7: Setup web interface
        setup_web_interface(config)

        # Step 8: Optional crypto
        setup_crypto_optional(config)

        # Step 9: Create directories and files
        create_directories(config)
        create_session_files(config)

        # Step 10: Print summary
        print_summary(config)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup cancelled.{Colors.END}")
        print("Run this script again to complete setup.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error during setup: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
