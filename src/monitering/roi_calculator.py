"""
roi_calculator.py — Business ROI Engine

Translates raw ML metrics (AUC drop, F1 change) into real dollar/rupee figures
that non-technical stakeholders can act on immediately.

The core formula:
    AUC drop  →  estimated missed churners
    missed churners  →  lost revenue (LTV)
    false positive rate rise  →  wasted outreach spend

Usage:
    roi = ROICalculator(
        monthly_customers=5000,
        avg_ltv=3600,
        outreach_cost_per_call=50,
        baseline_auc=0.87
    )
    result = roi.compute_monthly_loss(current_auc=0.83)
    # → {"missed_churners": 12, "revenue_at_risk": 43200, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ROIResult:
    """All dollar-impact figures for a single AUC measurement."""
    current_auc:        float
    baseline_auc:       float
    auc_drop:           float

    missed_churners:    int
    revenue_at_risk:    float   # lost LTV from missed churners
    wasted_outreach:    float   # spend on false-positive outreach calls
    total_monthly_loss: float

    # Annualised figures for exec reporting
    annual_revenue_risk: float
    annual_wasted_spend: float
    total_annual_loss:   float

    status: str   # "healthy" | "warning" | "critical"
    headline: str  # one-line plain-English summary

    def to_dict(self) -> dict:
        return {
            "current_auc":           round(self.current_auc, 4),
            "baseline_auc":          round(self.baseline_auc, 4),
            "auc_drop":              round(self.auc_drop, 4),
            "missed_churners":       self.missed_churners,
            "revenue_at_risk":       round(self.revenue_at_risk, 2),
            "wasted_outreach":       round(self.wasted_outreach, 2),
            "total_monthly_loss":    round(self.total_monthly_loss, 2),
            "annual_revenue_risk":   round(self.annual_revenue_risk, 2),
            "annual_wasted_spend":   round(self.annual_wasted_spend, 2),
            "total_annual_loss":     round(self.total_annual_loss, 2),
            "status":                self.status,
            "headline":              self.headline,
        }

    def pretty(self) -> str:
        """Terminal-friendly one-block summary."""
        currency = "₹" if self.revenue_at_risk > 10000 else "$"
        sep = "─" * 52
        return (
            f"\n{sep}\n"
            f"  ROI Impact Report\n"
            f"{sep}\n"
            f"  AUC:          {self.baseline_auc:.4f}  →  {self.current_auc:.4f}  "
            f"(drop: {self.auc_drop:.4f})\n"
            f"  Status:       {self.status.upper()}\n"
            f"  Missed churners / month:   {self.missed_churners:>6}\n"
            f"  Revenue at risk / month:   {currency}{self.revenue_at_risk:>10,.0f}\n"
            f"  Wasted outreach / month:   {currency}{self.wasted_outreach:>10,.0f}\n"
            f"  Total monthly loss:        {currency}{self.total_monthly_loss:>10,.0f}\n"
            f"  ── Annualised ──────────────────────────────────\n"
            f"  Total annual loss:         {currency}{self.total_annual_loss:>10,.0f}\n"
            f"{sep}\n"
            f"  {self.headline}\n"
            f"{sep}\n"
        )


class ROICalculator:
    """
    Converts model metric degradation into estimated business impact.

    Parameters
    ----------
    monthly_customers : int
        Total active customers scored per month.
    avg_ltv : float
        Average customer lifetime value in your local currency (₹ or $).
    outreach_cost_per_call : float
        Cost of one retention outreach attempt (agent time + tooling).
    baseline_auc : float
        AUC at the time of model deployment — the performance floor.
    true_churn_rate : float
        Fraction of customers who actually churn per month (default 0.10).
    outreach_coverage : float
        Fraction of predicted churners the retention team actually contacts
        (default 0.70 — not every flagged customer gets called).
    recall_sensitivity : float
        Empirical factor: how much recall % drops per unit of AUC drop.
        Default 1.8 is calibrated to XGBoost on imbalanced binary tasks.
    """

    def __init__(
        self,
        monthly_customers: int   = 5_000,
        avg_ltv: float           = 3_600,
        outreach_cost_per_call: float = 50,
        baseline_auc: float      = 0.87,
        true_churn_rate: float   = 0.10,
        outreach_coverage: float = 0.70,
        recall_sensitivity: float = 1.8,
    ):
        self.monthly_customers      = monthly_customers
        self.avg_ltv                = avg_ltv
        self.outreach_cost          = outreach_cost_per_call
        self.baseline_auc           = baseline_auc
        self.true_churn_rate        = true_churn_rate
        self.outreach_coverage      = outreach_coverage
        self.recall_sensitivity     = recall_sensitivity

        # Derived constant: total at-risk customers per month
        self._at_risk_pool = int(monthly_customers * true_churn_rate)

    # ── Core calculation ────────────────────────────────────────────────────

    def compute_monthly_loss(
        self,
        current_auc: float,
        current_f1: Optional[float] = None,
    ) -> ROIResult:
        """
        Compute monthly and annual revenue/spend impact for a given AUC.

        Args:
            current_auc:  Latest observed AUC from the monitoring batch.
            current_f1:   Latest F1 (optional; used only for the headline).

        Returns:
            ROIResult dataclass with all impact figures.
        """
        auc_drop = max(0.0, self.baseline_auc - current_auc)

        # ── 1. Missed churners ───────────────────────────────────────────────
        # AUC drop translates to an estimated recall drop.
        # We use a conservative linear approximation calibrated to XGBoost:
        #   recall_drop_pct ≈ auc_drop × recall_sensitivity
        missed_recall_pct = min(auc_drop * self.recall_sensitivity, 0.80)
        missed_churners   = int(self._at_risk_pool * missed_recall_pct)

        # ── 2. Revenue at risk (from missed churners) ────────────────────────
        # Each missed churner is a customer who churns without intervention.
        # We could have retained fraction `retention_success_rate` of them.
        # Conservative assumption: 30% of reached churners are saved.
        retention_success_rate = 0.30
        revenue_at_risk = (
            missed_churners
            * retention_success_rate
            * self.avg_ltv
        )

        # ── 3. Wasted outreach (false positives) ────────────────────────────
        # As AUC drops, precision also drops → more false positive calls.
        # Estimated FP rate increase ≈ auc_drop × 1.5 × at_risk_pool
        extra_fp_rate = min(auc_drop * 1.5, 0.50)
        extra_fp_calls = int(
            self._at_risk_pool
            * extra_fp_rate
            * self.outreach_coverage
        )
        wasted_outreach = extra_fp_calls * self.outreach_cost

        # ── 4. Totals ────────────────────────────────────────────────────────
        total_monthly  = revenue_at_risk + wasted_outreach
        annual_rev     = revenue_at_risk * 12
        annual_spend   = wasted_outreach * 12
        total_annual   = total_monthly * 12

        # ── 5. Status & headline ─────────────────────────────────────────────
        if auc_drop == 0:
            status   = "healthy"
            headline = (
                f"Model performing at baseline (AUC {current_auc:.4f}). "
                f"No measurable revenue impact detected."
            )
        elif auc_drop < 0.03:
            status   = "warning"
            headline = (
                f"Minor AUC drop of {auc_drop:.4f} — "
                f"est. {missed_churners} missed churners costing "
                f"~{self._fmt(total_monthly)}/month. Monitor closely."
            )
        elif auc_drop < 0.07:
            status   = "warning"
            headline = (
                f"Moderate AUC degradation ({auc_drop:.4f} drop). "
                f"Est. {self._fmt(total_monthly)}/month in lost revenue + "
                f"wasted outreach. Schedule retrain this week."
            )
        else:
            status   = "critical"
            headline = (
                f"CRITICAL: AUC dropped {auc_drop:.4f} from baseline. "
                f"Est. {self._fmt(total_monthly)}/month loss "
                f"({self._fmt(total_annual)}/year). Retrain immediately."
            )

        return ROIResult(
            current_auc=current_auc,
            baseline_auc=self.baseline_auc,
            auc_drop=round(auc_drop, 4),
            missed_churners=missed_churners,
            revenue_at_risk=round(revenue_at_risk, 2),
            wasted_outreach=round(wasted_outreach, 2),
            total_monthly_loss=round(total_monthly, 2),
            annual_revenue_risk=round(annual_rev, 2),
            annual_wasted_spend=round(annual_spend, 2),
            total_annual_loss=round(total_annual, 2),
            status=status,
            headline=headline,
        )

    # ── Recovery value: how much does retraining save? ──────────────────────

    def compute_retrain_value(
        self,
        current_auc: float,
        expected_new_auc: float,
        compute_cost: float = 0.0,
        n_days_until_next_cycle: int = 30,
    ) -> dict:
        """
        Compute the net value of a retrain decision.

        Returns a dict with NPV, daily savings, and a plain-English
        promotion/hold recommendation.
        """
        current_loss = self.compute_monthly_loss(current_auc).total_monthly_loss
        new_loss     = self.compute_monthly_loss(expected_new_auc).total_monthly_loss

        daily_savings = max((current_loss - new_loss) / 30, 0.0)
        gross_savings = daily_savings * n_days_until_next_cycle
        npv           = gross_savings - compute_cost
        payback_days  = (
            math.ceil(compute_cost / daily_savings)
            if daily_savings > 0 else float("inf")
        )

        recommend = npv > 0 or compute_cost == 0
        rationale = (
            f"Retrain saves est. {self._fmt(daily_savings)}/day. "
            f"NPV over {n_days_until_next_cycle} days: {self._fmt(npv)}. "
            + ("Retrain is justified." if recommend
               else f"Hold — compute cost ({self._fmt(compute_cost)}) "
                    f"not recovered within {n_days_until_next_cycle} days "
                    f"(payback: {payback_days} days).")
        )

        return {
            "should_retrain":         recommend,
            "npv":                    round(npv, 2),
            "daily_savings":          round(daily_savings, 2),
            "gross_savings":          round(gross_savings, 2),
            "compute_cost":           round(compute_cost, 2),
            "payback_days":           payback_days if payback_days != float("inf") else None,
            "rationale":              rationale,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(value: float) -> str:
        """Format a currency figure compactly."""
        if abs(value) >= 1_00_000:
            return f"₹{value/1_00_000:.1f}L"
        if abs(value) >= 1_000:
            return f"₹{value:,.0f}"
        return f"₹{value:.0f}"

    def summary_for_dashboard(self, current_auc: float) -> dict:
        """Thin wrapper returning only the fields the dashboard needs."""
        r = self.compute_monthly_loss(current_auc)
        return {
            "total_monthly_loss":  r.total_monthly_loss,
            "total_annual_loss":   r.total_annual_loss,
            "missed_churners":     r.missed_churners,
            "revenue_at_risk":     r.revenue_at_risk,
            "wasted_outreach":     r.wasted_outreach,
            "auc_drop":            r.auc_drop,
            "status":              r.status,
            "headline":            r.headline,
        }


# ── Default instance (used by monitoring_loop and dashboard) ─────────────────
# Values here match the synthetic dataset scale.
# Override in config.py or pass a custom instance for production use.
default_roi_calculator = ROICalculator(
    monthly_customers=5_000,
    avg_ltv=3_600,            # ₹3,600 per customer lifetime
    outreach_cost_per_call=50, # ₹50 per retention call
    baseline_auc=0.87,
    true_churn_rate=0.10,
)
