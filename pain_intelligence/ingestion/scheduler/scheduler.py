"""APScheduler integration for running ingestion on interval/cron schedules."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.config import IngestionConfig
from pain_intelligence.ingestion.engine import IngestionEngine
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class IngestionScheduler:
    """APScheduler wrapper that triggers IngestionEngine.run().

    Decoupled from the engine — the scheduler simply calls engine.run()
    on configured intervals. The engine knows nothing about APScheduler.
    """

    def __init__(self, engine: IngestionEngine, config: IngestionConfig) -> None:
        self._engine = engine
        self._config = config
        self._scheduler: Any = None

    def run_once(self, sources: list[str] | None = None) -> dict[str, Any]:
        """Run all (or specified) collectors once and return results."""
        logger.info("Running ingestion (one-shot)")
        return self._engine.run(sources=sources)

    def start(self) -> None:
        """Start the APScheduler with configured intervals.

        Schedule types:
        - 'once': Run once and stop
        - 'hourly': Run every hour
        - 'daily': Run once per day at 2:00 AM UTC
        - 'weekly': Run once per week on Sunday at 2:00 AM UTC
        """
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler = BlockingScheduler()
        schedule = self._config.schedule

        if schedule == "once":
            logger.info("Schedule is 'once' — running immediately")
            self._engine.run()
            return

        elif schedule == "hourly":
            trigger = IntervalTrigger(hours=1)
            logger.info("Schedule: hourly")

        elif schedule == "daily":
            trigger = CronTrigger(hour=2, minute=0)
            logger.info("Schedule: daily at 02:00 UTC")

        elif schedule == "weekly":
            trigger = CronTrigger(day_of_week="sun", hour=2, minute=0)
            logger.info("Schedule: weekly on Sunday at 02:00 UTC")

        else:
            logger.warning("Unknown schedule '{}', defaulting to daily", schedule)
            trigger = CronTrigger(hour=2, minute=0)

        self._scheduler.add_job(self._engine.run, trigger, id="ingestion_job")
        logger.info("Scheduler started. Press Ctrl+C to stop.")

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
            self.stop()

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down")
