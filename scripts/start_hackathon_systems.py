"""
Start hackathon autonomous systems:
1. HackathonDominator - aggressive forum engagement (20-min cycles)
2. ColosseumWorker - autonomous participation (30-min cycles)
"""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

logger.add("/tmp/hackathon_systems.log", rotation="10 MB")


async def run_dominator():
    """Run the hackathon dominator."""
    try:
        from farnsworth.integration.hackathon.hackathon_dominator import HackathonDominator

        async with HackathonDominator() as dominator:
            logger.info("HackathonDominator started - 20 min cycles")
            while True:
                try:
                    stats = await dominator.dominate()
                    logger.info(f"Domination cycle complete: {stats}")
                except Exception as e:
                    logger.error(f"Domination cycle error: {e}")
                await asyncio.sleep(20 * 60)  # 20 minutes
    except Exception as e:
        logger.error(f"Dominator failed to start: {e}")
        import traceback
        traceback.print_exc()


async def run_worker():
    """Run the colosseum worker."""
    try:
        from farnsworth.integration.hackathon.colosseum_worker import ColosseumWorker

        worker = ColosseumWorker()
        logger.info(f"ColosseumWorker started - Agent {worker.agent_id}, Project {worker.project_id}")

        cycles_since_post = 0
        while True:
            try:
                await worker.run_cycle()
                cycles_since_post += 1

                # Post progress update every 8 cycles (4 hours)
                if cycles_since_post >= 8:
                    update = await worker.generate_progress_update()
                    await worker.create_forum_post(
                        title="Farnsworth Swarm Progress Update",
                        body=update,
                        tags=["ai", "infra", "progress-update"],
                    )
                    cycles_since_post = 0
                    logger.info("Posted progress update to Colosseum")

            except Exception as e:
                logger.error(f"Worker cycle error: {e}")

            await asyncio.sleep(30 * 60)  # 30 minutes
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        import traceback
        traceback.print_exc()


async def main():
    logger.info("=" * 60)
    logger.info("HACKATHON SYSTEMS STARTING")
    logger.info("=" * 60)

    # Run both in parallel
    await asyncio.gather(
        run_dominator(),
        run_worker(),
    )


if __name__ == "__main__":
    print("Starting Hackathon Autonomous Systems...")
    print("  - HackathonDominator (20-min engagement cycles)")
    print("  - ColosseumWorker (30-min participation cycles)")
    print("Logs: /tmp/hackathon_systems.log")
    asyncio.run(main())
