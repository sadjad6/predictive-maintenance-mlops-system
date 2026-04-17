"""What-If simulation engine for maintenance scenario analysis.

Allows users to simulate different maintenance strategies and
see their impact on failure probability and costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import BusinessConfig, get_config


@dataclass
class WhatIfScenario:
    """A maintenance what-if scenario result."""

    scenario_name: str
    maintenance_cycle: int
    current_failure_prob: float
    projected_failure_prob: float
    probability_change: float
    cost_of_maintenance: float
    cost_of_failure: float
    expected_savings: float
    is_recommended: bool
    explanation: str


class WhatIfSimulator:
    """Simulates maintenance timing scenarios.

    Models the impact of early/delayed maintenance on failure
    probability and financial outcomes.
    """

    def __init__(self, config: BusinessConfig | None = None) -> None:
        self.config = config or get_config().business

    def simulate_maintenance_timing(
        self,
        current_cycle: int,
        current_failure_prob: float,
        rul_estimate: float,
        maintenance_cycles: list[int] | None = None,
    ) -> list[WhatIfScenario]:
        """Simulate multiple maintenance timing scenarios.

        Args:
            current_cycle: Current operating cycle.
            current_failure_prob: Current failure probability.
            rul_estimate: Estimated remaining useful life.
            maintenance_cycles: List of candidate maintenance times.

        Returns:
            List of WhatIfScenario results, one per candidate.
        """
        if maintenance_cycles is None:
            # Generate default scenarios at 25%, 50%, 75%, 100% of RUL
            maintenance_cycles = [
                current_cycle + int(rul_estimate * pct) for pct in [0.25, 0.50, 0.75, 1.0]
            ]

        scenarios = []
        for maint_cycle in maintenance_cycles:
            scenario = self._evaluate_scenario(
                current_cycle,
                current_failure_prob,
                rul_estimate,
                maint_cycle,
            )
            scenarios.append(scenario)

        return scenarios

    def _evaluate_scenario(
        self,
        current_cycle: int,
        current_prob: float,
        rul: float,
        maintenance_cycle: int,
    ) -> WhatIfScenario:
        """Evaluate a single maintenance timing scenario."""
        cycles_until_maintenance = max(1, maintenance_cycle - current_cycle)
        cycles_until_failure = max(1, rul)

        # Failure probability increases as we approach end of life
        fraction_of_life_used = min(1.0, cycles_until_maintenance / cycles_until_failure)

        # Exponential risk increase model
        projected_prob_no_maint = min(
            0.99,
            current_prob * np.exp(fraction_of_life_used * 1.5),
        )

        # Post-maintenance: risk drops to ~10% of pre-maintenance level
        post_maint_prob = current_prob * 0.1

        # Cost analysis
        failure_cost = self.config.downtime_cost_per_hour * self.config.avg_repair_hours
        expected_failure_cost = projected_prob_no_maint * failure_cost
        maint_cost = self.config.maintenance_cost
        expected_savings = expected_failure_cost - maint_cost

        is_recommended = expected_savings > 0
        timing_desc = _timing_description(cycles_until_maintenance)

        return WhatIfScenario(
            scenario_name=f"Maintenance at cycle {maintenance_cycle}",
            maintenance_cycle=maintenance_cycle,
            current_failure_prob=round(current_prob, 4),
            projected_failure_prob=round(post_maint_prob, 4),
            probability_change=round(current_prob - post_maint_prob, 4),
            cost_of_maintenance=round(maint_cost, 2),
            cost_of_failure=round(expected_failure_cost, 2),
            expected_savings=round(expected_savings, 2),
            is_recommended=is_recommended,
            explanation=(
                f"Performing maintenance {timing_desc} "
                f"({'saves' if is_recommended else 'costs'} "
                f"${abs(expected_savings):,.0f}). "
                f"Risk drops from {current_prob:.1%} to {post_maint_prob:.1%}."
            ),
        )


def _timing_description(cycles: int) -> str:
    """Convert cycles to human-readable timing."""
    if cycles < 10:
        return "immediately"
    if cycles < 50:
        return f"in {cycles} cycles (~{cycles} hours)"
    return f"in {cycles} cycles (~{cycles // 24} days)"
