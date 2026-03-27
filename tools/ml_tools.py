"""ML pipeline tools for the Engineer and Builder agents — regression version."""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from config import MODELS_DIR


# ── Feature engineering helpers ──────────────────────────────────────────────

def _extract_date_features(df: pd.DataFrame, date_col: str = "last_dt") -> pd.DataFrame:
    """Parse last_dt into numeric date features; drop the original string column."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    reference = pd.Timestamp("2019-12-31")          # ~end of dataset window
    df["days_since_last_review"] = (reference - parsed).dt.days
    df["last_review_year"]  = parsed.dt.year
    df["last_review_month"] = parsed.dt.month
    df.drop(columns=[date_col], inplace=True)
    return df


def _drop_id_cols(df: pd.DataFrame,
                  id_cols: list[str] | None = None) -> pd.DataFrame:
    """Drop high-cardinality text identifier columns."""
    default_ids = ["name", "_id", "host_name"]
    to_drop = [c for c in (id_cols or default_ids) if c in df.columns]
    return df.drop(columns=to_drop)


def _prepare_X(df: pd.DataFrame,
               target_col: str,
               drop_cols: list[str] | None = None) -> pd.DataFrame:
    """Return feature matrix after date extraction and ID dropping."""
    drop = list(drop_cols or [])
    if target_col in df.columns:
        drop.append(target_col)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = _extract_date_features(X)
    X = _drop_id_cols(X)
    return X


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    ohe_cols = [c for c in cat_cols if X[c].nunique() <= 30]
    ord_cols = [c for c in cat_cols if X[c].nunique() > 30]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    ohe_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ord_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if ohe_cols:
        transformers.append(("ohe", ohe_pipe, ohe_cols))
    if ord_cols:
        transformers.append(("ord", ord_pipe, ord_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Main tools class ──────────────────────────────────────────────────────────

class MLTools:
    """Regression pipeline utilities exposed as agent tools."""

    @staticmethod
    def prepare_features(path: str, target_col: str,
                          drop_cols: list[str] | None = None) -> dict[str, Any]:
        """Summarise feature types after preprocessing."""
        df = pd.read_csv(path)
        X = _prepare_X(df, target_col, drop_cols)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return {
            "feature_columns": list(X.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "n_samples": len(df),
            "n_features": len(X.columns),
        }

    @staticmethod
    def train_and_evaluate(path: str, target_col: str,
                            model_name: str = "random_forest",
                            drop_cols: list[str] | None = None,
                            test_size: float = 0.2,
                            cv_folds: int = 5) -> dict[str, Any]:
        """Train a regressor and return CV + hold-out metrics (MSE, RMSE, MAE, R2)."""
        df = pd.read_csv(path)
        X = _prepare_X(df, target_col, drop_cols)
        y = df[target_col]

        preprocessor = _build_preprocessor(X)

        model_map = {
            "ridge":             Ridge(alpha=1.0),
            "random_forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                           max_depth=5, random_state=42),
        }
        reg = model_map.get(model_name, model_map["random_forest"])
        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        holdout = {
            "mse":  round(float(mean_squared_error(y_te, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
            "mae":  round(float(mean_absolute_error(y_te, y_pred)), 4),
            "r2":   round(float(r2_score(y_te, y_pred)), 4),
        }

        # CV on MSE (neg_mean_squared_error → positive)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_neg_mse = cross_val_score(pipeline, X, y, cv=kf,
                                      scoring="neg_mean_squared_error")
        cv_mse_scores = -cv_neg_mse

        model_path = MODELS_DIR / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        return {
            "model": model_name,
            "holdout_metrics": holdout,
            "cv_mse_mean":  round(float(cv_mse_scores.mean()), 4),
            "cv_mse_std":   round(float(cv_mse_scores.std()), 4),
            "cv_rmse_mean": round(float(np.sqrt(cv_mse_scores.mean())), 4),
            "model_path": str(model_path),
        }

    @staticmethod
    def compare_models(path: str, target_col: str,
                        drop_cols: list[str] | None = None) -> dict[str, Any]:
        """Train and compare all regressors; rank by CV MSE (lower = better)."""
        models = ["ridge", "random_forest", "gradient_boosting"]
        results = {}
        for m in models:
            try:
                res = MLTools.train_and_evaluate(
                    path, target_col, model_name=m, drop_cols=drop_cols
                )
                results[m] = {
                    "cv_mse":  res["cv_mse_mean"],
                    "cv_std":  res["cv_mse_std"],
                    "cv_rmse": res["cv_rmse_mean"],
                    "holdout": res["holdout_metrics"],
                }
            except Exception as e:
                results[m] = {"error": str(e)}

        ranked = sorted(
            [(m, v) for m, v in results.items() if "error" not in v],
            key=lambda x: x[1]["cv_mse"],
        )
        return {
            "results": results,
            "ranking": [m for m, _ in ranked],
            "best_model": ranked[0][0] if ranked else None,
        }

    @staticmethod
    def feature_importance(path: str, target_col: str,
                            drop_cols: list[str] | None = None,
                            top_n: int = 15) -> dict[str, Any]:
        """Feature importances from a Random Forest regressor."""
        df = pd.read_csv(path)
        X = _prepare_X(df, target_col, drop_cols)
        y = df[target_col]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        ohe_cols = [c for c in cat_cols if X[c].nunique() <= 30]
        ord_cols = [c for c in cat_cols if X[c].nunique() > 30]

        preprocessor = _build_preprocessor(X)
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])
        pipeline.fit(X, y)

        importances = pipeline.named_steps["reg"].feature_importances_
        try:
            ohe = pipeline.named_steps["pre"].named_transformers_["ohe"].named_steps["enc"]
            ohe_feat_names = ohe.get_feature_names_out(ohe_cols).tolist()
        except Exception:
            ohe_feat_names = []
        all_names = num_cols + ohe_feat_names + ord_cols

        imp_df = pd.DataFrame({
            "feature":    all_names[:len(importances)],
            "importance": importances[:len(all_names)],
        }).sort_values("importance", ascending=False).head(top_n)
        return {"feature_importances": imp_df.to_dict(orient="records")}

    @staticmethod
    def generate_submission(train_path: str, test_path: str,
                             target_col: str, model_name: str = "gradient_boosting",
                             drop_cols: list[str] | None = None,
                             output_path: str = "submission.csv") -> dict[str, Any]:
        """
        Train on full train set and predict test set.
        Saves submission in sample_submition.csv format: index, prediction.
        """
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        X_train = _prepare_X(train_df, target_col, drop_cols)
        y_train = train_df[target_col]
        X_test  = _prepare_X(test_df, target_col, drop_cols)

        # Align columns: test may lack some OHE-generated columns
        # handled by handle_unknown="ignore" in OHE, so just fit on train
        preprocessor = _build_preprocessor(X_train)
        model_map = {
            "ridge":             Ridge(alpha=1.0),
            "random_forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                           max_depth=5, random_state=42),
        }
        reg = model_map.get(model_name, model_map["gradient_boosting"])
        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        # Clip to valid range [0, 365]
        preds = np.clip(preds, 0, 365)

        submission = pd.DataFrame({
            "index":      range(len(preds)),
            "prediction": preds,
        })
        submission.to_csv(output_path, index=False)

        # Also save pipeline
        model_path = MODELS_DIR / f"{model_name}_full.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        return {
            "submission_path": output_path,
            "n_predictions": len(preds),
            "pred_mean":  round(float(preds.mean()), 4),
            "pred_std":   round(float(preds.std()), 4),
            "pred_min":   round(float(preds.min()), 4),
            "pred_max":   round(float(preds.max()), 4),
            "model_path": str(model_path),
        }

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        return [
            {
                "name": "prepare_features",
                "description": "Summarise feature types after date extraction and ID dropping.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "train_and_evaluate",
                "description": "Train a regressor with CV and return MSE/RMSE/MAE/R2.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "enum": ["ridge", "random_forest", "gradient_boosting"],
                            "default": "random_forest",
                        },
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                        "test_size": {"type": "number", "default": 0.2},
                        "cv_folds":  {"type": "integer", "default": 5},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "compare_models",
                "description": "Compare Ridge, Random Forest, and Gradient Boosting by CV MSE.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "feature_importance",
                "description": "Compute feature importances using Random Forest.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                        "top_n": {"type": "integer", "default": 15},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "generate_submission",
                "description": "Train on full train set, predict test set, save submission CSV.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "train_path":  {"type": "string"},
                        "test_path":   {"type": "string"},
                        "target_col":  {"type": "string"},
                        "model_name":  {"type": "string", "default": "gradient_boosting"},
                        "drop_cols":   {"type": "array", "items": {"type": "string"}, "default": []},
                        "output_path": {"type": "string", "default": "submission.csv"},
                    },
                    "required": ["train_path", "test_path", "target_col"],
                },
            },
        ]

    def dispatch(self, tool_name: str, tool_input: dict) -> Any:
        fn = getattr(self, tool_name, None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**tool_input)
        except Exception as exc:
            return {"error": str(exc)}
