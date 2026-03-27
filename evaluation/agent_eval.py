"""
Agent effectiveness benchmarking.

Measures:
  - Tool call efficiency (calls made / useful results)
  - Critique loop convergence (how many rounds needed)
  - Plan adherence (steps executed vs planned)
  - Decision quality (did Critic approve on first pass?)
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EXPERIMENTS_DIR
from memory.experiment_store import ExperimentStore


class AgentEvaluator:
    """Compute agent-level benchmarks from a completed run."""

    def __init__(self, store: ExperimentStore):
        self.store = store

    def evaluate(self) -> dict[str, Any]:
        """
        Produce a benchmark report for the current run.
        """
        events = self.store.get_events()
        model_events = self.store.get_events("model_result")
        critique_events = self.store.get_events("critique")
        plan_events = self.store.get_events("plan")
        feature_events = self.store.get_events("feature_decisions")

        # ── Metric 1: Critique convergence ──────────────────────────
        total_critiques = len(critique_events)
        first_pass_approvals = sum(
            1 for e in critique_events
            if e["payload"].get("approved") is True
        )
        convergence_rate = (
            round(first_pass_approvals / total_critiques, 2)
            if total_critiques else None
        )

        # ── Metric 2: Model improvement across iterations ────────────
        auc_scores = []
        for e in model_events:
            auc = e["payload"].get("cv_roc_auc_mean")
            if auc is not None:
                auc_scores.append(float(auc))
        best_auc = max(auc_scores) if auc_scores else None
        auc_improvement = (
            round(auc_scores[-1] - auc_scores[0], 4)
            if len(auc_scores) >= 2 else None
        )

        # ── Metric 3: Plan adherence ─────────────────────────────────
        planned_steps = 0
        if plan_events:
            plan = plan_events[0]["payload"].get("plan", [])
            planned_steps = len(plan)
        executed_agent_calls = len(set(e["agent"] for e in events
                                       if e["agent"] != "system"))

        # ── Metric 4: Agent participation ────────────────────────────
        agent_counts: dict[str, int] = {}
        for e in events:
            a = e["agent"]
            agent_counts[a] = agent_counts.get(a, 0) + 1

        report = {
            "run_id": self.store.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "agent_participation": agent_counts,
            "critique": {
                "total_critiques": total_critiques,
                "first_pass_approvals": first_pass_approvals,
                "convergence_rate": convergence_rate,
            },
            "models": {
                "experiments_run": len(model_events),
                "auc_trajectory": auc_scores,
                "best_cv_roc_auc": best_auc,
                "auc_improvement": auc_improvement,
            },
            "planning": {
                "planned_steps": planned_steps,
                "unique_agents_used": executed_agent_calls,
            },
        }
        # Save report
        report_path = EXPERIMENTS_DIR / self.store.run_id / "agent_benchmark.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    @staticmethod
    def format_report(report: dict) -> str:
        lines = [
            "## Agent Benchmark Report",
            f"Run ID: `{report['run_id']}`",
            "",
            "### Critique Loop",
            f"- Total critiques issued: {report['critique']['total_critiques']}",
            f"- First-pass approvals: {report['critique']['first_pass_approvals']}",
            f"- Convergence rate: {report['critique']['convergence_rate']}",
            "",
            "### Model Performance",
            f"- Experiments run: {report['models']['experiments_run']}",
            f"- Best CV ROC-AUC: {report['models']['best_cv_roc_auc']}",
            f"- AUC improvement: {report['models']['auc_improvement']}",
            "",
            "### Agent Participation",
        ]
        for agent, count in report.get("agent_participation", {}).items():
            lines.append(f"- {agent}: {count} events")
        return "\n".join(lines)
