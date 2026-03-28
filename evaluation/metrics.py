"""Regression model quality metrics."""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelMetrics:
    """Compute and format a regression evaluation report."""

    @staticmethod
    def evaluate_saved_model(model_path: str, dataset_path: str,
                              target_col: str) -> dict[str, Any]:
        """Load a saved sklearn pipeline and evaluate on an 80/20 hold-out split.

        Applies only base preprocessing (date extraction, ID dropping) without
        engineer-decided drops, so the saved pipeline can select its own features
        via its fitted ColumnTransformer (remainder='drop').
        """
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        from tools.ml_tools import _prepare_X
        df = pd.read_csv(dataset_path)
        # Pass drop_cols=None: only target + identifiers are removed.
        # The pipeline's ColumnTransformer uses only the columns it was trained on.
        X = _prepare_X(df, target_col, drop_cols=None)
        y = df[target_col]

        from sklearn.model_selection import train_test_split
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = pipeline.predict(X_te)
        return ModelMetrics.compute(y_te, y_pred)

    @staticmethod
    def compute(y_true, y_pred) -> dict[str, Any]:
        """Compute MSE, RMSE, MAE, R2."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse":  round(mse, 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
            "r2":   round(float(r2_score(y_true, y_pred)), 4),
        }

    @staticmethod
    def format_report(metrics: dict) -> str:
        return "\n".join([
            "## Model Evaluation Report",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| MSE    | {metrics.get('mse', '---')} |",
            f"| RMSE   | {metrics.get('rmse', '---')} |",
            f"| MAE    | {metrics.get('mae', '---')} |",
            f"| R2     | {metrics.get('r2', '---')} |",
        ])
