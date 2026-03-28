"""
Builder agent — model selection, training, and evaluation.
"""
import json

from agents.base_agent import BaseAgent
from config import MODELS
from tools.ml_tools import MLTools


class BuilderAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["builder"], **kwargs)
        self.name = "Builder"
        self.role = (
            "You are an ML engineer specialised in tabular regression. "
            "The task is to MINIMISE MSE on a rental occupancy prediction problem (target: days 0-365). "
            "Use the ML tools to compare regressors (Ridge, Random Forest, Gradient Boosting), "
            "select the best model by cross-validated MSE, and produce an evaluation report. "
            "Output your final recommendation as JSON with keys: "
            "'chosen_model', 'reasoning', 'metrics', 'drop_columns'."
        )
        self._ml = MLTools()
        self._dispatchers = [self._ml]
        self.tools = MLTools.get_tool_definitions()

    def build(self, dataset_path: str, target_col: str,
               feature_decisions: dict, critic_feedback: str = "") -> dict:
        """
        Compare models, select the best, run final evaluation.
        Returns the result dict from the best model.
        """
        drop_cols = feature_decisions.get("drop_columns", [])

        prompt = (
            f"Dataset: {dataset_path}\n"
            f"Target: {target_col} (regression, minimise MSE)\n"
            f"Drop columns: {drop_cols}\n\n"
            "1. Call compare_models to benchmark Ridge, Random Forest, Gradient Boosting.\n"
            "2. Call train_and_evaluate with the best model for detailed MSE/RMSE/MAE/R2.\n"
            "3. Call feature_importance to understand which features matter.\n"
        )
        if critic_feedback:
            prompt += f"\nCritic feedback:\n{critic_feedback}\n"

        prompt += (
            "\nAfter calling the tools, output a JSON recommendation with keys: "
            "chosen_model, reasoning, metrics, drop_columns."
        )

        raw = self.run(
            prompt,
            rag_query="regression model selection LightGBM gradient boosting MSE KFold cross-validation hyperparameters tabular",
        )
        recommendation = self._parse_recommendation(raw)

        # Always run the definitive train_and_evaluate for the chosen model
        final_result = MLTools.train_and_evaluate(
            dataset_path,
            target_col,
            model_name=recommendation.get("chosen_model", "gradient_boosting"),
            drop_cols=drop_cols,
        )
        final_result["recommendation"] = recommendation
        self.store.log_model_result(final_result, agent=self.name)
        return final_result

    @staticmethod
    def _parse_recommendation(raw: str) -> dict:
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            # Default to random forest
            return {
                "chosen_model": "random_forest",
                "reasoning": raw[:300],
                "metrics": {},
                "drop_columns": [],
            }
