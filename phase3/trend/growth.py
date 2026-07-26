"""Growth, velocity, and momentum analyzers for trend detection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class GrowthAnalyzer:
    """Computes growth percentage, velocity, and acceleration between snapshots."""

    def compute(
        self,
        current_value: float,
        prior_value: float,
        days_elapsed: float = 1.0,
    ) -> dict[str, float]:
        """Compute growth metrics between two data points.

        Args:
            current_value: Value in the latest snapshot.
            prior_value: Value in the prior snapshot.
            days_elapsed: Number of days between the two snapshots.

        Returns:
            Dict with growth_pct, velocity, and acceleration.
        """
        if prior_value == 0 and current_value == 0:
            return {"growth_pct": 0.0, "velocity": 0.0, "acceleration": 0.0}

        growth_pct = 0.0
        if prior_value > 0:
            growth_pct = ((current_value - prior_value) / prior_value) * 100.0
        elif current_value > 0:
            growth_pct = 100.0

        velocity = (current_value - prior_value) / max(days_elapsed, 0.1)

        return {
            "growth_pct": round(growth_pct, 4),
            "velocity": round(velocity, 4),
            "acceleration": 0.0,
        }

    def compute_acceleration(
        self,
        current_velocity: float,
        prior_velocity: float,
        days_elapsed: float = 1.0,
    ) -> float:
        """Compute acceleration (change in velocity over time)."""
        if days_elapsed <= 0:
            return 0.0
        return round((current_velocity - prior_velocity) / days_elapsed, 4)


class VelocityAnalyzer:
    """Analyzes the rate of change (velocity) across multiple snapshots."""

    def compute(
        self,
        values: list[float],
        timestamps: list[str],
    ) -> dict[str, float]:
        """Compute velocity metrics from a series of values.

        Args:
            values: Values at each snapshot (ordered oldest to newest).
            timestamps: ISO-8601 timestamps for each value.

        Returns:
            Dict with avg_velocity, peak_velocity, momentum.
        """
        if len(values) < 2:
            return {"avg_velocity": 0.0, "peak_velocity": 0.0, "momentum": 0.0}

        velocities: list[float] = []
        for i in range(1, len(values)):
            days = _days_between(timestamps[i - 1], timestamps[i])
            if days <= 0:
                days = 1.0
            vel = (values[i] - values[i - 1]) / days
            velocities.append(vel)

        avg_vel = sum(velocities) / len(velocities) if velocities else 0.0
        peak_vel = max(velocities) if velocities else 0.0

        # Momentum: weighted average favoring recent
        momentum = self._compute_momentum(velocities)

        return {
            "avg_velocity": round(avg_vel, 4),
            "peak_velocity": round(peak_vel, 4),
            "momentum": round(momentum, 4),
        }

    def _compute_momentum(self, velocities: list[float]) -> float:
        if not velocities:
            return 0.0
        weights = [(i + 1) / len(velocities) for i in range(len(velocities))]
        weighted = sum(v * w for v, w in zip(velocities, weights))
        total_w = sum(weights)
        return weighted / total_w if total_w > 0 else 0.0


class MomentumAnalyzer:
    """Computes momentum as a normalized [0, 1] score from trend metrics."""

    def compute(
        self,
        growth_pct: float,
        velocity: float,
        acceleration: float,
        snapshot_count: int,
    ) -> float:
        """Compute normalized momentum score.

        Momentum combines growth strength, velocity, and acceleration,
        normalized to [0, 1].
        """
        growth_norm = min(abs(growth_pct) / 200.0, 1.0)
        vel_norm = min(abs(velocity) / 1000.0, 1.0)
        accel_norm = min(abs(acceleration) / 100.0, 1.0)
        snap_factor = min(snapshot_count / 10.0, 1.0)

        return round(
            (growth_norm * 0.4 + vel_norm * 0.3 + accel_norm * 0.2 + snap_factor * 0.1),
            4,
        )


def _days_between(t1: str, t2: str) -> float:
    try:
        dt1 = datetime.fromisoformat(t1)
        dt2 = datetime.fromisoformat(t2)
        return abs((dt2 - dt1).total_seconds() / 86400.0)
    except Exception:
        return 1.0
