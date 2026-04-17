"""KPI engine: translates ML predictions into business metrics.

Computes cost savings, downtime reduction, ROI, and maintenance
scheduling recommendations from model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.config import BusinessConfig, get_config


@dataclass
class BusinessKPIs:
    """Business KPI summary for executive reporting."""

    total_machines: int = 0
    machines_at_risk: int = 0
    avg_health_score: float = 0.0
    estimated_annual_savings: float = 0.0
    downtime_hours_prevented: float = 0.0
    maintenance_events_scheduled: int = 0
    cost_per_prevented_failure: float = 0.0
    roi_percentage: float = 0.0
    risk_distribution: dict[str, int] = field(default_factory=dict)


class KPIEngine:
    """Computes business KPIs from prediction outputs.

    Translates failure probabilities, RUL estimates, and anomaly
    scores into actionable business metrics with configurable
    cost assumptions.
    """

    def __init__(self, config: BusinessConfig | None = None) -> None:
        self.config = config or get_config().business

    def compute_kpis(
        self,
        failure_probs: np.ndarray,
        rul_estimates: np.ndarray | None = None,
        annual_operating_hours: float = 8760.0,
    ) -> BusinessKPIs:
        """Compute business KPIs from model predictions.

        Args:
            failure_probs: Array of failure probabilities per machine.
            rul_estimates: Optional array of RUL estimates.
            annual_operating_hours: Total operating hours per year.

        Returns:
            BusinessKPIs with financial and operational metrics.
        """
        kpis = BusinessKPIs()
        kpis.total_machines = len(failure_probs)

        # Risk classification
        kpis.risk_distribution = {
            "critical": int(np.sum(failure_probs >= 0.8)),
            "high": int(np.sum((failure_probs >= 0.6) & (failure_probs < 0.8))),
            "medium": int(np.sum((failure_probs >= 0.3) & (failure_probs < 0.6))),
            "low": int(np.sum(failure_probs < 0.3)),
        }
        kpis.machines_at_risk = kpis.risk_distribution["critical"] + kpis.risk_distribution["high"]

        # Health score: inverse of average failure probability
        kpis.avg_health_score = round(float(1.0 - np.mean(failure_probs)) * 100, 1)

        # Cost calculations
        expected_failures = float(np.sum(failure_probs))
        prevented_failures = expected_failures * 0.85  # 85% detection rate assumption

        failure_cost = (
            self.config.downtime_cost_per_hour * self.config.avg_repair_hours
        )
        maintenance_cost = self.config.maintenance_cost

        kpis.estimated_annual_savings = round(
            prevented_failures * (failure_cost - maintenance_cost), 2,
        )
        kpis.downtime_hours_prevented = round(
            prevented_failures * self.config.avg_repair_hours, 1,
        )
        kpis.maintenance_events_scheduled = int(np.ceil(prevented_failures))

        if prevented_failures > 0:
            kpis.cost_per_prevented_failure = round(
                maintenance_cost + (failure_cost * 0.1), 2,  # 10% partial downtime
            )

        # ROI (assuming $50K annual system cost)
        system_cost = 50_000.0
        if system_cost > 0:
            kpis.roi_percentage = round(
                (kpis.estimated_annual_savings - system_cost) / system_cost * 100, 1,
            )

        logger.info(
            "KPIs: {} machines, {} at risk, savings=${:,.0f}, ROI={:.1f}%",
            kpis.total_machines, kpis.machines_at_risk,
            kpis.estimated_annual_savings, kpis.roi_percentage,
        )
        return kpis

    def maintenance_schedule(
        self,
        engine_ids: list[int],
        failure_probs: np.ndarray,
        rul_estimates: np.ndarray,
    ) -> list[dict]:
        """Generate prioritized maintenance schedule.

        Returns:
            List of dicts with engine_id, priority, and recommended timing.
        """
        schedule = []
        for i, engine_id in enumerate(engine_ids):
            prob = float(failure_probs[i])
            rul = float(rul_estimates[i]) if rul_estimates is not None else 999

            if prob >= 0.8 or rul < 10:
                priority, timing = "CRITICAL", "Within 24 hours"
            elif prob >= 0.6 or rul < 30:
                priority, timing = "HIGH", "Within 3 days"
            elif prob >= 0.3 or rul < 60:
                priority, timing = "MEDIUM", "Within 2 weeks"
            else:
                priority, timing = "LOW", "Next scheduled maintenance"

            schedule.append({
                "engine_id": engine_id,
                "failure_probability": round(prob, 4),
                "estimated_rul": round(rul, 1),
                "priority": priority,
                "recommended_timing": timing,
                "estimated_cost_if_missed": round(
                    prob * self.config.downtime_cost_per_hour * self.config.avg_repair_hours, 2,
                ),
            })

        # Sort by priority (critical first)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        schedule.sort(key=lambda x: priority_order.get(x["priority"], 4))
        return schedule
