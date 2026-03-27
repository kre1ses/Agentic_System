"""
Critic agent — reviews outputs from Explorer, Engineer, and Builder
and returns structured feedback.
"""
import json

from agents.base_agent import BaseAgent
from config import MODELS


class CriticAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["critic"], **kwargs)
        self.name = "Critic"
        self.role = (
            "You are a rigorous ML peer reviewer. "
            "Evaluate the provided work for correctness, best practices, "
            "potential data leakage, overfitting risks, and missed opportunities. "
            "Output a JSON object with keys: "
            "'approved' (bool), 'severity' ('ok'|'minor'|'major'), "
            "'issues' (list of strings), 'suggestions' (list of strings). "
            "Be constructive and specific."
        )

    def review_eda(self, eda_report: dict) -> dict:
        """Review an EDA report."""
        prompt = (
            "Review the following EDA report for completeness and correctness.\n\n"
            f"{json.dumps(eda_report, default=str)[:3000]}\n\n"
            "Focus on: missing-value handling, class imbalance, feature leakage risk. "
            "Output ONLY valid JSON."
        )
        return self._critique(
            prompt, "eda review class imbalance missing values leakage"
        )

    def review_feature_decisions(self, decisions: dict, eda_report: dict) -> dict:
        """Review the Engineer's feature decisions."""
        prompt = (
            "Review these feature engineering decisions given the EDA report.\n\n"
            f"Decisions: {json.dumps(decisions, default=str)}\n\n"
            f"EDA summary (missingness + correlations):\n"
            f"{json.dumps({'missing': eda_report.get('missing', {}), 'correlations': eda_report.get('correlations', {})}, default=str)[:2000]}\n\n"
            "Check for: dropped important features, encoding errors, leakage. "
            "Output ONLY valid JSON."
        )
        return self._critique(
            prompt, "feature engineering leakage encoding best practices"
        )

    def review_model_results(self, model_result: dict) -> dict:
        """Review the Builder's model results."""
        prompt = (
            "Review these regression model training results (task: minimise MSE).\n\n"
            f"{json.dumps(model_result, default=str)[:2000]}\n\n"
            "Check for: overfitting (holdout vs CV gap), high RMSE relative to target range 0-365, "
            "data leakage, missing date feature extraction, unexploited features. "
            "Output ONLY valid JSON."
        )
        return self._critique(
            prompt, "regression MSE overfitting cross-validation feature engineering"
        )

    def _critique(self, prompt: str, rag_query: str) -> dict:
        raw = self.run(prompt, rag_query=rag_query)
        result = self._parse_critique(raw)
        self.store.log_critique(result, agent=self.name)
        return result

    @staticmethod
    def _parse_critique(raw: str) -> dict:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            # If LLM didn't produce JSON, treat as minor feedback
            return {
                "approved": True,
                "severity": "minor",
                "issues": [],
                "suggestions": [raw[:400]],
            }
