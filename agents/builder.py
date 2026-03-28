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
            "Available models: " + ", ".join(MLTools.AVAILABLE_MODELS) + ". "
            "Workflow: "
            "1. Call compare_models to benchmark all available models. "
            "2. Call train_and_evaluate with the best model for detailed metrics. "
            "3. Call feature_importance on the best model. "
            "Output your final recommendation as JSON with keys: "
            "'chosen_model', 'reasoning', 'metrics', 'drop_columns'."
        )
        self._ml = MLTools()
        self._dispatchers = [self._ml]
        self.tools = MLTools.get_tool_definitions()

    def build(self, dataset_path: str, target_col: str,
               feature_decisions: dict, critic_feedback: str = "") -> dict:
        """
        Compare models, select the best (or ensemble), run final evaluation.
        Returns the result dict from the best approach.
        """
        drop_cols = feature_decisions.get("drop_columns", [])

        prompt = (
            f"Dataset: {dataset_path}\n"
            f"Target: {target_col} (regression, minimise MSE — rental occupancy days 0-365)\n"
            f"Drop columns: {drop_cols}\n"
            f"Available models: {MLTools.AVAILABLE_MODELS}\n\n"
            "1. Call compare_models to benchmark all available models by CV MSE.\n"
            "2. Call train_and_evaluate with the best model.\n"
            "3. Call feature_importance on the best model.\n"
        )
        if critic_feedback:
            prompt += f"\nCritic feedback from previous iteration:\n{critic_feedback}\n"

        prompt += (
            "\nOutput a JSON recommendation with keys: "
            "chosen_model, reasoning, metrics, drop_columns."
        )

        raw = self.run(
            prompt,
            rag_query="regression model selection LightGBM gradient boosting MSE KFold cross-validation hyperparameters tabular",
        )
        recommendation = self._parse_recommendation(raw)

        model_name = recommendation.get("chosen_model", "lightgbm")
        final_result = MLTools.train_and_evaluate(
            dataset_path, target_col,
            model_name=model_name,
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
            # Default to lightgbm if available
            from tools.ml_tools import _LGB_AVAILABLE
            default_model = "lightgbm" if _LGB_AVAILABLE else "gradient_boosting"
            return {
                "chosen_model":  default_model,
                "reasoning":     raw[:300],
                "metrics":       {},
                "drop_columns":  [],
            }
