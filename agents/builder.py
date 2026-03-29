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
            "The task is to MINIMISE MSE on a rental occupancy prediction problem "
            "(target: days 0-365, many listings may have 0 occupied days). "
            "Available models: " + ", ".join(MLTools.AVAILABLE_MODELS) + ". "
            "\n\nSPECIAL MODEL TYPES:\n"
            "  'ensemble'  — trains LightGBM + XGBoost + CatBoost and averages their "
            "predictions; usually beats any single model.\n"
            "  'two_stage' — trains a binary classifier (is target > 0?) followed by "
            "a regressor on positive-only rows; ideal when >15 % of targets are zero.\n"
            "\nWORKFLOW:\n"
            "1. Call prepare_features to check target_zero_fraction and "
            "recommend_two_stage.\n"
            "2. Call compare_models to benchmark all available models by CV MSE. "
            "Always include 'ensemble' and 'two_stage' in the comparison.\n"
            "3. If the best model from step 2 is a single tunable model (not 'ensemble' "
            "or 'two_stage'), call tune_hyperparameters on it with n_trials=40 to find "
            "optimal hyperparameters via Optuna Bayesian search.\n"
            "4. Call train_and_evaluate with the best model from step 2 (for final "
            "confirmation; skip if tune_hyperparameters already returned holdout metrics).\n"
            "5. Call feature_importance on the best model.\n"
            "\nOutput your final recommendation as JSON with keys: "
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
            "Step 1. Call prepare_features to check target zero-fraction and "
            "whether a two-stage model is recommended.\n"
            "Step 2. Call compare_models — always include 'ensemble' and 'two_stage' "
            "to benchmark the composite approaches alongside single models.\n"
            "Step 3. If the winner is a tunable single model (not ensemble/two_stage), "
            "call tune_hyperparameters with n_trials=40 to find optimal hyperparameters.\n"
            "Step 4. Call train_and_evaluate with the best model (skip if step 3 ran).\n"
            "Step 5. Call feature_importance on the best model.\n"
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
            rec = json.loads(raw[start:end])
            # Ensure chosen_model is a valid option
            from tools.ml_tools import MLTools
            if rec.get("chosen_model") not in MLTools.AVAILABLE_MODELS:
                from tools.ml_tools import _LGB_AVAILABLE
                rec["chosen_model"] = "lightgbm" if _LGB_AVAILABLE else "gradient_boosting"
            return rec
        except Exception:
            from tools.ml_tools import _LGB_AVAILABLE
            default_model = "lightgbm" if _LGB_AVAILABLE else "gradient_boosting"
            return {
                "chosen_model":  default_model,
                "reasoning":     raw[:300],
                "metrics":       {},
                "drop_columns":  [],
            }
