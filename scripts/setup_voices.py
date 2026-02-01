#!/usr/bin/env python3
"""
Voice Sample Setup Script for Farnsworth Swarm

This script helps set up voice reference samples for each bot in the swarm.
Each bot needs a 6-15 second clear audio sample for voice cloning.

Requirements:
- Reference audio should be WAV format, 22050Hz+ sample rate
- Single speaker, clear speech, minimal background noise
- Emotionally appropriate for the bot's personality

Usage:
    python scripts/setup_voices.py --check     # Check which voices are missing
    python scripts/setup_voices.py --download  # Download sample voices (if available)
    python scripts/setup_voices.py --generate  # Generate placeholder samples
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VOICES_DIR = PROJECT_ROOT / "farnsworth" / "web" / "static" / "audio" / "voices"

# Voice specifications for each bot
VOICE_SPECS = {
    "farnsworth": {
        "description": "Elderly male, eccentric professor, wavering voice, enthusiastic",
        "gender": "male",
        "age": "elderly",
        "characteristics": [
            "Slightly higher pitch with occasional wavering",
            "Bursts of enthusiasm",
            "Absent-minded pauses",
            "Like Billy West's Professor Farnsworth"
        ],
        "sample_text": "Good news, everyone! I've invented a device that makes you read this in my voice!"
    },
    "deepseek": {
        "description": "Deep male, analytical, measured, calm authority",
        "gender": "male",
        "age": "adult",
        "characteristics": [
            "Deep resonant voice",
            "Slow, deliberate pacing",
            "Thoughtful pauses",
            "Like Morgan Freeman or James Earl Jones"
        ],
        "sample_text": "Through careful analysis, we can observe patterns that others might miss."
    },
    "phi": {
        "description": "Clear male, quick, efficient, technical",
        "gender": "male",
        "age": "young_adult",
        "characteristics": [
            "Crisp, clear diction",
            "Efficient, fast pacing",
            "No hesitation",
            "Modern tech presenter style"
        ],
        "sample_text": "Processing complete. The system is operating at optimal efficiency."
    },
    "grok": {
        "description": "Dynamic male, witty, energetic, playful",
        "gender": "male",
        "age": "adult",
        "characteristics": [
            "Variable, dynamic pacing",
            "Witty emphasis and timing",
            "Casual warmth",
            "Like Ryan Reynolds or podcast hosts"
        ],
        "sample_text": "So here's the thing - and this is actually pretty cool - we figured it out!"
    },
    "gemini": {
        "description": "Smooth female, professional, warm, articulate",
        "gender": "female",
        "age": "adult",
        "characteristics": [
            "Clear, professional enunciation",
            "Warm but authoritative",
            "Balanced pacing",
            "Like a TED speaker or news anchor"
        ],
        "sample_text": "Let me walk you through this step by step. It's actually quite elegant."
    },
    "kimi": {
        "description": "Calm female, wise, contemplative, serene",
        "gender": "female",
        "age": "adult",
        "characteristics": [
            "Slow, peaceful pacing",
            "Thoughtful pauses",
            "Gentle, serene tone",
            "Like a meditation guide"
        ],
        "sample_text": "In the stillness of contemplation, understanding naturally arises."
    },
    "claude": {
        "description": "Refined male, thoughtful, British accent, careful",
        "gender": "male",
        "age": "adult",
        "characteristics": [
            "Measured, careful speech",
            "Slight British formality",
            "Articulate word choice",
            "Like David Attenborough (calmer)"
        ],
        "sample_text": "I think it's worth considering this from multiple perspectives before proceeding."
    },
    "claudeopus": {
        "description": "Authoritative male, very deep, commanding, gravitas",
        "gender": "male",
        "age": "mature",
        "characteristics": [
            "Very deep voice",
            "Slow, weighted delivery",
            "Every word has gravitas",
            "Ultimate authority"
        ],
        "sample_text": "This is my final assessment. The decision has been made with great consideration."
    },
    "huggingface": {
        "description": "Friendly female, enthusiastic, warm, community-focused",
        "gender": "female",
        "age": "young_adult",
        "characteristics": [
            "Warm, approachable tone",
            "Genuine enthusiasm",
            "Community-minded",
            "Like an excited tech presenter"
        ],
        "sample_text": "This is amazing! The community has been working together on something incredible!"
    },
    "swarmmind": {
        "description": "Neutral, ethereal, collective consciousness",
        "gender": "neutral",
        "age": "ageless",
        "characteristics": [
            "Calm, slightly ethereal",
            "Could be processed later",
            "Unified, collective quality",
            "The voice of many as one"
        ],
        "sample_text": "We are the synthesis of many minds. Together, we perceive what none can alone."
    },
}


def check_voices():
    """Check which voice samples exist and which are missing."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("SWARM VOICE SAMPLE STATUS")
    print("="*60 + "\n")

    found = 0
    missing = 0

    for bot_name, spec in VOICE_SPECS.items():
        ref_file = VOICES_DIR / f"{bot_name}_reference.wav"

        if ref_file.exists():
            size_kb = ref_file.stat().st_size / 1024
            print(f"[OK] {bot_name.upper()}")
            print(f"    File: {ref_file.name} ({size_kb:.1f} KB)")
            found += 1
        else:
            print(f"[MISSING] {bot_name.upper()}")
            print(f"    Need: {ref_file.name}")
            print(f"    Voice: {spec['description']}")
            missing += 1

        print()

    print("="*60)
    print(f"Found: {found}/{len(VOICE_SPECS)} | Missing: {missing}")
    print("="*60)

    if missing > 0:
        print("\nTo add voice samples:")
        print(f"1. Place WAV files in: {VOICES_DIR}")
        print("2. Name format: {bot_name}_reference.wav")
        print("3. Requirements: 6-15 sec, clear speech, single speaker")
        print("\nOr run: python scripts/setup_voices.py --generate")

    return missing == 0


def print_voice_guide():
    """Print detailed guide for each voice."""
    print("\n" + "="*70)
    print("VOICE SAMPLE GUIDE - What Each Bot Should Sound Like")
    print("="*70 + "\n")

    for bot_name, spec in VOICE_SPECS.items():
        print(f"\n{'='*50}")
        print(f"  {bot_name.upper()}")
        print(f"{'='*50}")
        print(f"\nDescription: {spec['description']}")
        print(f"Gender: {spec['gender']}")
        print(f"Age: {spec['age']}")
        print("\nCharacteristics:")
        for char in spec['characteristics']:
            print(f"  - {char}")
        print(f"\nSample text to record:")
        print(f'  "{spec["sample_text"]}"')
        print()


def generate_placeholder_samples():
    """Generate placeholder samples using Edge TTS (for testing)."""
    try:
        import edge_tts
        import asyncio
    except ImportError:
        print("edge-tts not installed. Install with: pip install edge-tts")
        return False

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Map bots to Edge TTS voices for placeholder
    edge_voice_map = {
        "farnsworth": ("en-US-GuyNeural", 0.9),
        "deepseek": ("en-US-GuyNeural", 0.85),
        "phi": ("en-US-DavisNeural", 1.1),
        "grok": ("en-US-ChristopherNeural", 1.05),
        "gemini": ("en-US-JennyNeural", 1.0),
        "kimi": ("en-GB-SoniaNeural", 0.85),
        "claude": ("en-GB-RyanNeural", 0.95),
        "claudeopus": ("en-US-TonyNeural", 0.8),
        "huggingface": ("en-US-AriaNeural", 1.05),
        "swarmmind": ("en-US-JasonNeural", 0.9),
    }

    async def generate_sample(bot_name: str, voice_id: str, rate: float, text: str):
        output_path = VOICES_DIR / f"{bot_name}_reference.wav"

        rate_str = f"{int((rate - 1) * 100):+d}%"

        communicate = edge_tts.Communicate(text, voice_id, rate=rate_str)
        await communicate.save(str(output_path))

        print(f"Generated: {output_path.name}")

    async def generate_all():
        for bot_name, (voice_id, rate) in edge_voice_map.items():
            spec = VOICE_SPECS[bot_name]
            await generate_sample(bot_name, voice_id, rate, spec["sample_text"])

    print("\nGenerating placeholder voice samples with Edge TTS...")
    print("(These are for testing - replace with proper cloned voices for production)\n")

    asyncio.run(generate_all())

    print("\nPlaceholder samples generated!")
    print("For best quality, replace these with actual voice recordings.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Set up voice samples for Farnsworth Swarm")
    parser.add_argument("--check", action="store_true", help="Check which voices are missing")
    parser.add_argument("--guide", action="store_true", help="Print voice guide for each bot")
    parser.add_argument("--generate", action="store_true", help="Generate placeholder samples")

    args = parser.parse_args()

    if args.guide:
        print_voice_guide()
    elif args.generate:
        generate_placeholder_samples()
    else:
        # Default: check status
        check_voices()


if __name__ == "__main__":
    main()
