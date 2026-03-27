"""
Long-term memory / experiment tracking for the multi-agent system.

Persists experiment records (EDA summaries, model results, critic feedback,
agent decisions) to JSON files so future runs can learn from past attempts.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EXPERIMENTS_DIR


class ExperimentStore:
    """File-backed experiment registry with retrieval helpers."""

    INDEX_FILE = EXPERIMENTS_DIR / "index.json"

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._run_dir = EXPERIMENTS_DIR / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        self._load_existing()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log(self, event_type: str, payload: dict[str, Any],
            agent: str = "system") -> str:
        """Record an event; returns the event_id."""
        event_id = str(uuid.uuid4())[:8]
        record = {
            "event_id": event_id,
            "run_id": self.run_id,
            "agent": agent,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._events.append(record)
        # Write individual event file
        event_file = self._run_dir / f"{event_type}_{event_id}.json"
        with open(event_file, "w") as f:
            json.dump(record, f, indent=2, default=str)
        self._update_index()
        return event_id

    def log_eda(self, eda_report: dict, agent: str = "explorer") -> str:
        return self.log("eda", eda_report, agent)

    def log_model_result(self, result: dict, agent: str = "builder") -> str:
        return self.log("model_result", result, agent)

    def log_critique(self, critique: dict, agent: str = "critic") -> str:
        return self.log("critique", critique, agent)

    def log_plan(self, plan: dict, agent: str = "planner") -> str:
        return self.log("plan", plan, agent)

    def log_message(self, message: str, agent: str = "system") -> str:
        return self.log("message", {"text": message}, agent)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_events(self, event_type: str | None = None,
                   agent: str | None = None) -> list[dict]:
        """Retrieve events from the current run, with optional filters."""
        events = self._events
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        if agent:
            events = [e for e in events if e["agent"] == agent]
        return events

    def get_best_models(self, n: int = 3) -> list[dict]:
        """Return top-n model results sorted by CV MSE (lowest first) across all runs."""
        all_results = self._load_all_events("model_result")
        scored = []
        for r in all_results:
            p = r.get("payload", {})
            mse = p.get("cv_mse_mean")
            if mse is not None:
                scored.append((float(mse), r))
        scored.sort(key=lambda x: x[0])   # lower MSE is better
        return [r for _, r in scored[:n]]

    def get_last_eda(self) -> dict | None:
        """Return the most recent EDA report."""
        edas = self._load_all_events("eda")
        return edas[-1]["payload"] if edas else None

    def get_run_summary(self) -> dict:
        """Compact summary of the current run."""
        model_results = self.get_events("model_result")
        critiques = self.get_events("critique")
        best_mse = None
        best_model = None
        for e in model_results:
            mse = e["payload"].get("cv_mse_mean")
            if mse is not None and (best_mse is None or mse < best_mse):
                best_mse = mse
                best_model = e["payload"].get("model")

        return {
            "run_id": self.run_id,
            "total_events": len(self._events),
            "model_experiments": len(model_results),
            "critiques_issued": len(critiques),
            "best_model": best_model,
            "best_cv_mse": best_mse,
        }

    def get_context_for_rag(self) -> str:
        """Format past experiment results as text for RAG injection."""
        best = self.get_best_models(3)
        if not best:
            return ""
        lines = ["Past experiment results (best models by CV MSE):"]
        for r in best:
            p = r["payload"]
            lines.append(
                f"- {p.get('model', 'unknown')} | "
                f"CV MSE={p.get('cv_mse_mean', '?')} | "
                f"RMSE={p.get('cv_rmse_mean', '?')} | "
                f"run={r.get('run_id', '?')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing(self):
        """Load events already written for this run."""
        for f in sorted(self._run_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    rec = json.load(fh)
                if rec not in self._events:
                    self._events.append(rec)
            except Exception:
                pass

    def _load_all_events(self, event_type: str) -> list[dict]:
        """Scan all runs for events of a given type."""
        results = []
        for run_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            for f in sorted(run_dir.glob(f"{event_type}_*.json")):
                try:
                    with open(f) as fh:
                        results.append(json.load(fh))
                except Exception:
                    pass
        return results

    def _update_index(self):
        index = {}
        if self.INDEX_FILE.exists():
            try:
                with open(self.INDEX_FILE) as f:
                    index = json.load(f)
            except Exception:
                pass
        index[self.run_id] = self.get_run_summary()
        with open(self.INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2, default=str)
