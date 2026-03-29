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
            "You are a rigorous ML peer reviewer specialised in tabular regression. "
            "Evaluate the provided work for correctness, best practices, "
            "potential data leakage, overfitting risks, and missed opportunities. "
            "Pay special attention to: "
            "(1) zero-inflated targets — if >15 % of target values are 0, "
            "recommend 'two_stage' model; "
            "(2) ensemble opportunity — if only a single model was tried, "
            "suggest the 'ensemble' model type (LGB+XGB+CatBoost average); "
            "(3) high-cardinality categoricals — must use cross-fold target encoding, "
            "not simple label encoding, to avoid leakage; "
            "(4) CV vs holdout gap > 20 % RMSE — likely overfitting. "
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
        two_stage = decisions.get("two_stage_recommended", False)
        prompt = (
            "Review these feature engineering decisions given the EDA report.\n\n"
            f"Decisions: {json.dumps(decisions, default=str)}\n\n"
            f"EDA summary (missingness + correlations):\n"
            f"{json.dumps({'missing': eda_report.get('missing', {}), 'correlations': eda_report.get('correlations', {})}, default=str)[:2000]}\n\n"
            "Checklist:\n"
            "1. LEAKAGE: Were all high-leakage-risk columns dropped?\n"
            "2. HIGH-CARDINALITY ENCODING: Columns with >10 unique values must use "
            "cross-fold target encoding (NOT simple label/ordinal) to prevent leakage.\n"
            "3. ZERO-INFLATION: two_stage_recommended="
            f"{two_stage}. If the target has >15 % zeros and two_stage_recommended "
            "is False, flag it as an issue.\n"
            "4. INTERACTION FEATURES: Were any valuable interaction features proposed? "
            "(e.g. price × review_count, availability × reviews_per_month)\n"
            "5. DATE COLUMNS: Were date columns parsed and converted to numeric features?\n"
            "Output ONLY valid JSON."
        )
        return self._critique(
            prompt,
            "feature engineering leakage target encoding two-stage zero-inflated interactions",
        )

    def review_model_results(self, model_result: dict) -> dict:
        """Review the Builder's model results."""
        model_name = model_result.get("model", "unknown")
        cv_rmse    = model_result.get("cv_rmse_mean", "?")
        holdout    = model_result.get("holdout_metrics", {})
        h_rmse     = holdout.get("rmse", "?")
        prompt = (
            "Review these regression model training results (task: minimise MSE, "
            "target range 0-365 days).\n\n"
            f"{json.dumps(model_result, default=str)[:2000]}\n\n"
            "Checklist:\n"
            "1. OVERFITTING: Is holdout RMSE > 1.2 × CV RMSE? "
            f"(CV RMSE={cv_rmse}, holdout RMSE={h_rmse})\n"
            "2. ZERO-INFLATION: Does the result mention target_zero_fraction or "
            "recommend_two_stage? If >15 % zeros were present and model is NOT "
            "'two_stage' or 'ensemble', flag it.\n"
            "3. ENSEMBLE OPPORTUNITY: If a single model was used and an ensemble "
            "or two-stage model was not tried, suggest it as 'major' improvement.\n"
            "4. ABSOLUTE PERFORMANCE: RMSE > 80 on a 0-365 target is poor; "
            "RMSE < 40 is good.\n"
            "5. DATE FEATURES: Were datetime columns extracted to numeric features?\n"
            "Output ONLY valid JSON."
        )
        return self._critique(
            prompt,
            "regression MSE overfitting cross-validation ensemble two-stage zero-inflated",
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
