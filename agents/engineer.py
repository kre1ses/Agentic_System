"""
Engineer agent — feature engineering and preprocessing decisions.
"""
import json

from agents.base_agent import BaseAgent
from config import MODELS
from tools.ml_tools import MLTools


class EngineerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["engineer"], **kwargs)
        self.name = "Engineer"
        self.role = (
            "You are a senior ML feature engineer. "
            "Given an EDA report, decide: "
            "(1) which columns to drop (high missingness, identifiers, leakage), "
            "(2) encoding strategy per categorical feature, "
            "(3) numeric transformations (log, binning, scaling), "
            "(4) any interaction features worth creating. "
            "Output your decisions as a JSON object with keys: "
            "'drop_columns', 'encode_columns', 'scale_columns', "
            "'log_transform_columns', 'notes'. "
            "Be concise and justify each decision."
        )
        self._ml = MLTools()
        self._dispatchers = [self._ml]
        self.tools = MLTools.get_tool_definitions()[:1]  # prepare_features

    def plan_features(self, eda_report: dict, dataset_path: str,
                       target_col: str, critic_feedback: str = "",
                       validation_report: dict | None = None) -> dict:
        """
        Given EDA report (and optional critic feedback / validation report),
        produce feature decisions.  Returns a dict of decisions.
        """
        eda_summary = json.dumps({
            "missing": eda_report.get("missing", {}),
            "correlations": eda_report.get("correlations", {}),
            "outliers": eda_report.get("outliers", {}),
            "feature_recs": eda_report.get("feature_recs", {}),
            "class_balance": eda_report.get("class_balance", {}),
        }, default=str)[:3000]

        prompt = (
            f"Dataset: {dataset_path}\n"
            f"Target: {target_col}\n\n"
            f"EDA summary:\n{eda_summary}\n\n"
        )

        if validation_report:
            val_context = {
                "drop_candidates": validation_report.get("drop_candidates", []),
                "leakage_suspects": [
                    s["column"] for s in validation_report.get("leakage_suspects", [])
                    if s.get("severity") == "high_leakage_risk"
                ],
                "high_missing": [
                    col for col, pct in validation_report.get("missingness_profile", {}).items()
                    if pct >= 0.8
                ],
                "target_skew": validation_report.get("target_stats", {}).get("skewness"),
                "recommended_actions": validation_report.get("recommended_actions", []),
            }
            prompt += (
                f"Validation report (pre-computed, trust these findings):\n"
                f"{json.dumps(val_context, default=str)}\n\n"
                "The drop_candidates MUST be included in drop_columns unless you have "
                "a strong reason to keep them. Leakage suspects must also be dropped.\n\n"
            )

        if critic_feedback:
            prompt += f"Critic feedback from previous iteration:\n{critic_feedback}\n\n"

        prompt += (
            "Decide on the feature engineering strategy. "
            "Also call prepare_features to confirm column types. "
            "Output ONLY valid JSON with keys: "
            "drop_columns, encode_columns, scale_columns, "
            "log_transform_columns, notes."
        )

        raw = self.run(
            prompt,
            rag_query="regression feature engineering target encoding log transform missing imputation ID columns leakage rental occupancy",
        )
        decisions = self._parse_decisions(raw)
        self.store.log(
            "feature_decisions",
            {"decisions": decisions, "feedback": critic_feedback,
             "validation_drop_candidates": (validation_report or {}).get("drop_candidates", [])},
            agent=self.name,
        )
        return decisions

    @staticmethod
    def _parse_decisions(raw: str) -> dict:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {
                "drop_columns": [],
                "encode_columns": [],
                "scale_columns": [],
                "log_transform_columns": [],
                "notes": raw[:500],
            }
