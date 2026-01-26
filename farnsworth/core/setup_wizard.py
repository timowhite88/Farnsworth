"""
Farnsworth Setup Wizard - Granular Configuration.

"I've built a machine that lets you choose your own fate!"

This module provides a step-by-step interactive setup for all Farnsworth systems.
It saves configuration to a local .env file.
"""

import os
import json
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import questionary # We'll assume this or use standard input fallbacks
from loguru import logger

class SetupWizard:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_file = project_root / ".env"
        self.config: Dict[str, str] = {}

    def _ask(self, question: str, default: str = "") -> str:
        """Standard input fallback if questionary isn't available."""
        val = input(f"❓ {question} [{default}]: ").strip()
        return val if val else default

    def _ask_bool(self, question: str, default: bool = True) -> bool:
        default_str = "Y/n" if default else "y/N"
        val = input(f"❓ {question} ({default_str}): ").strip().lower()
        if not val: return default
        return val.startswith('y')

    def _ask_select(self, question: str, options: List[str], default: str) -> str:
        print(f"\nOptions: {', '.join(options)}")
        return self._ask(question, default)

    async def run(self):
        print("\n" + "="*50)
        print("🚀 FARNSWORTH GRANULAR SETUP WIZARD")
        print("="*50 + "\n")

        # 1. Privacy & Networking
        print("🛡️  PRIVACY & NETWORKING")
        is_isolated = self._ask_bool("Enable ISOLATED MODE? (Disables P2P swarm and network broadcasting)", default=False)
        self.config["FARNSWORTH_ISOLATED"] = "true" if is_isolated else "false"
        
        if not is_isolated:
            self.config["FARNSWORTH_P2P_PORT"] = self._ask("P2P Swarm Port", "9999")
            self.config["FARNSWORTH_DISCOVERY_PORT"] = self._ask("UDP Discovery Port", "8888")

        # 2. Hardware Profile
        print("\n💻 HARDWARE PROFILE")
        profile = self._ask_select(
            "Select hardware profile:", 
            ["minimal", "cpu_only", "low_vram", "medium_vram", "high_vram"], 
            "medium_vram"
        )
        self.config["FARNSWORTH_HARDWARE_PROFILE"] = profile

        # 3. Cognitive Engines
        print("\n🧠 COGNITIVE ENGINES")
        self.config["ENABLE_TOM"] = "true" if self._ask_bool("Enable Theory of Mind (User simulation)?") else "false"
        self.config["ENABLE_CAUSAL"] = "true" if self._ask_bool("Enable Causal Reasoning?") else "false"
        self.config["ENABLE_VIDEO_FLOW"] = "true" if self._ask_bool("Enable Advanced Video Flow Analysis?") else "false"

        # 4. External Integrations
        print("\n🔗 EXTERNAL INTEGRATIONS")
        if self._ask_bool("Configure GITHUB integration?", default=False):
            self.config["GITHUB_TOKEN"] = self._ask("GitHub Personal Access Token")
            self.config["GITHUB_REPO"] = self._ask("Default Repository (user/repo)")

        if self._ask_bool("Configure X (TWITTER) integration?", default=False):
            self.config["X_API_KEY"] = self._ask("X API Key")
            self.config["X_API_SECRET"] = self._ask("X API Secret")
            self.config["X_ACCESS_TOKEN"] = self._ask("X Access Token")
            self.config["X_ACCESS_SECRET"] = self._ask("X Access Secret")

        if self._ask_bool("Configure OFFICE 365 integration?", default=False):
            self.config["O365_CLIENT_ID"] = self._ask("O365 Client ID")
            self.config["O365_CLIENT_SECRET"] = self._ask("O365 Client Secret")

        if self._ask_bool("Configure n8n integration?", default=False):
            self.config["N8N_WEBHOOK_URL"] = self._ask("n8n Webhook URL")

        # 5. Advanced Skills
        print("\n🚀 ADVANCED SKILLS")
        if self._ask_bool("Enable Grok (xAI) Integration?"):
            self.config["XAI_API_KEY"] = self._ask("xAI API Key")

        if self._ask_bool("Enable Remotion Video Generation?"):
            self.config["REMOTION_WORKSPACE"] = self._ask("Remotion Workspace Path", "./remotion_workspace")

        if self._ask_bool("Enable Discord Bridge (ChatOps)?"):
            self.config["DISCORD_TOKEN"] = self._ask("Discord Bot Token")

        if self._ask_bool("Configure SQL Database Access?"):
            self.config["DB_TYPE"] = self._ask_select("DB Type", ["sqlite", "postgres", "mysql"], "sqlite")
            self.config["DB_CONNECTION_STRING"] = self._ask("Connection String", "farnsworth_data.db")

        self.config["ENABLE_PARALLEL_AI"] = "true" if self._ask_bool("Enable Parallel AI (Multi-model consensus)?") else "false"

        # 6. Save Configuration
        self._save()
        print("\n" + "="*50)
        print("✅ SETUP COMPLETE!")
        print(f"Configuration saved to {self.env_file}")
        print("="*50 + "\n")

    def _save(self):
        with open(self.env_file, "w") as f:
            f.write("# Farnsworth Environment Configuration\n")
            for key, value in self.config.items():
                if value: # Only save non-empty values
                    f.write(f"{key}={value}\n")

if __name__ == "__main__":
    import asyncio
    wizard = SetupWizard(Path("."))
    asyncio.run(wizard.run())
