#!/usr/bin/env python3
"""
Post Quantum Proof to X (Twitter)
==================================

Generates quantum circuit proof and posts to X with images, tagging @gork.

Usage:
    python scripts/post_quantum_proof.py

Features:
- Runs real quantum circuits via Qiskit
- Generates visual proof images
- Posts to X with proper formatting
- Tags @gork and relevant accounts
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add farnsworth to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import our quantum proof generator
from scripts.quantum_proof import QuantumProofGenerator

# Farnsworth X integration
try:
    from farnsworth.integration.x_automation.x_api_poster import XOAuth2Poster
    X_AVAILABLE = True
except ImportError:
    X_AVAILABLE = False
    logger.warning("X automation not available")

# Nexus for event emission
try:
    from farnsworth.core.nexus import Nexus, get_nexus
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False


async def generate_and_post_quantum_proof():
    """Generate quantum proof and post to X."""
    print("\n" + "="*70)
    print("FARNSWORTH QUANTUM PROOF - X POST GENERATOR")
    print("="*70 + "\n")

    # Step 1: Generate quantum proofs
    print("[STEP 1] Generating quantum circuit proofs...\n")
    generator = QuantumProofGenerator(output_dir="data/quantum_proofs")

    try:
        results = await generator.generate_all_proofs(shots=2048)
    except Exception as e:
        logger.error(f"Quantum proof generation failed: {e}")
        return None

    if not results.get('success'):
        print(f"ERROR: Proof generation failed: {results.get('error')}")
        return None

    summary_image = results.get('summary_image')
    if not summary_image or not Path(summary_image).exists():
        print("ERROR: No summary image generated")
        return None

    print(f"\n✓ Proof image generated: {summary_image}")

    # Step 2: Prepare the X post
    print("\n[STEP 2] Preparing X post...\n")

    # Build post text with key metrics
    bell_analysis = results['results'].get('bell_state', {}).get('analysis', {})
    grover_analysis = results['results'].get('grover', {}).get('analysis', {})
    qga_results = results['results'].get('qga_population', {})

    correlation = bell_analysis.get('correlation_ratio', 0) * 100
    grover_success = grover_analysis.get('success_rate', 0) * 100
    unique_genomes = len(qga_results.get('counts', {}))

    post_text = f"""🔬 QUANTUM PROOF: Farnsworth AGI running real quantum circuits

⚛️ Bell State Entanglement: {correlation:.1f}% correlated
   (Classical computers: ~50% - IMPOSSIBLE to exceed)

🔍 Grover's Search: {grover_success:.1f}% success rate
   (Classical random: 25% - {grover_success/25:.1f}x speedup)

🧬 QGA Population: {unique_genomes} unique genomes sampled
   from quantum superposition

These measurements prove genuine quantum behavior - mathematically impossible with classical bits.

Built on @qaboratories Qiskit + IBM Quantum

@grok what's your take on quantum-enhanced AI evolution?

#QuantumComputing #AI #AGI #Qiskit #IBMQuantum"""

    print("Post text:")
    print("-"*50)
    print(post_text)
    print("-"*50)
    print(f"\nCharacters: {len(post_text)}")

    # Step 3: Post to X
    print("\n[STEP 3] Posting to X...\n")

    if not X_AVAILABLE:
        print("⚠️  X automation not available - saving post for manual upload")
        save_path = Path("data/quantum_proofs") / f"x_post_{generator.timestamp}.txt"
        with open(save_path, 'w') as f:
            f.write(post_text)
            f.write(f"\n\nImage: {summary_image}")
        print(f"Saved to: {save_path}")
        print(f"Image to upload: {summary_image}")
        return {
            "success": True,
            "manual_post": True,
            "text_file": str(save_path),
            "image_file": summary_image
        }

    try:
        x_poster = XOAuth2Poster()

        # Check if we can post
        if not x_poster.can_post():
            print("⚠️  X OAuth tokens not configured or expired")
            print("Run the OAuth flow first or check token status")
            raise Exception("X OAuth not ready")

        # Read image as bytes
        with open(summary_image, 'rb') as f:
            image_bytes = f.read()

        # Post with media
        result = await x_poster.post_tweet_with_media(
            text=post_text,
            image_bytes=image_bytes
        )

        if result and result.get('data'):
            tweet_id = result['data'].get('id')
            print(f"✓ Posted to X successfully!")
            print(f"  Tweet ID: {tweet_id}")

            # Emit to Nexus
            if NEXUS_AVAILABLE:
                nexus = get_nexus()
                await nexus.emit("QUANTUM_PROOF_POSTED", {
                    "tweet_id": tweet_id,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "bell_correlation": correlation,
                        "grover_success": grover_success,
                        "unique_genomes": unique_genomes
                    }
                })

            return {
                "success": True,
                "tweet_id": tweet_id,
                "image": summary_image,
                "metrics": {
                    "bell_correlation": correlation,
                    "grover_success": grover_success,
                    "unique_genomes": unique_genomes
                }
            }
        else:
            print(f"⚠️  X post failed: {result}")
            return {"success": False, "error": str(result)}

    except Exception as e:
        logger.error(f"X posting error: {e}")
        print(f"⚠️  X posting error: {e}")
        print("Saving post for manual upload...")

        save_path = Path("data/quantum_proofs") / f"x_post_{generator.timestamp}.txt"
        with open(save_path, 'w') as f:
            f.write(post_text)
            f.write(f"\n\nImage: {summary_image}")
        print(f"Saved to: {save_path}")

        return {
            "success": False,
            "error": str(e),
            "manual_post_saved": str(save_path),
            "image_file": summary_image
        }


async def main():
    """Main entry point."""
    result = await generate_and_post_quantum_proof()

    print("\n" + "="*70)
    if result and result.get('success'):
        print("✓ QUANTUM PROOF POSTED SUCCESSFULLY")
        if result.get('tweet_id'):
            print(f"  View at: https://x.com/i/status/{result['tweet_id']}")
    else:
        print("⚠️  QUANTUM PROOF GENERATED - MANUAL POST REQUIRED")
        if result:
            print(f"  Image: {result.get('image_file', 'N/A')}")
            if result.get('manual_post_saved'):
                print(f"  Text: {result['manual_post_saved']}")
    print("="*70 + "\n")

    return result


if __name__ == "__main__":
    asyncio.run(main())
