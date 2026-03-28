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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    _CB_AVAILABLE = True
except ImportError:
    _CB_AVAILABLE = False

from config import MODELS_DIR


# ── Feature engineering helpers ───────────────────────────────────────────────

def _extract_date_features(df: pd.DataFrame, date_col: str = "last_dt") -> pd.DataFrame:
    """
    Parse a date column into rich numeric features.
    Adds cyclical sin/cos for month and day_of_week, plus linear features.
    """
    df = df.copy()
    if date_col not in df.columns:
        return df
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    reference = pd.Timestamp("2019-12-31")
    df["days_since_last_review"] = (reference - parsed).dt.days
    df["last_review_year"]  = parsed.dt.year
    df["last_review_month"] = parsed.dt.month
    df["last_review_dow"]   = parsed.dt.dayofweek          # 0=Mon … 6=Sun
    df["last_review_quarter"] = parsed.dt.quarter
    df["last_review_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
    # Cyclical encoding
    df["last_review_month_sin"] = np.sin(2 * np.pi * parsed.dt.month / 12)
    df["last_review_month_cos"] = np.cos(2 * np.pi * parsed.dt.month / 12)
    df["last_review_dow_sin"]   = np.sin(2 * np.pi * parsed.dt.dayofweek / 7)
    df["last_review_dow_cos"]   = np.cos(2 * np.pi * parsed.dt.dayofweek / 7)
    df.drop(columns=[date_col], inplace=True)
    return df


def _add_missing_indicators(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    For each column with > threshold fraction of NaN, add a binary
    '<col>_was_missing' indicator.  GBM models can learn from missingness patterns.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().mean() > threshold:
            df[f"{col}_was_missing"] = df[col].isna().astype(np.int8)
    return df


def _add_group_features(df: pd.DataFrame, target_col: str | None = None,
                         group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Add group aggregation features: mean/std of numeric columns grouped by
    each categorical column.  Computed on the passed dataframe only (no leakage
    from test set when called on train only, then applied to test via merge).
    """
    df = df.copy()
    if group_cols is None:
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        group_cols = [c for c in cat_cols if 2 < df[c].nunique() <= 50]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Limit to top-3 most variable numeric cols to avoid explosion
    if numeric_cols:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(3).index.tolist()

    for g_col in group_cols[:3]:           # max 3 group cols
        for n_col in numeric_cols[:3]:     # max 3 numeric cols
            key_mean = f"{g_col}__{n_col}_mean"
            key_std  = f"{g_col}__{n_col}_std"
            agg = df.groupby(g_col)[n_col].agg(["mean", "std"]).rename(
                columns={"mean": key_mean, "std": key_std}
            )
            df = df.merge(agg, on=g_col, how="left")
    return df


def _target_encode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    high_card_cols: list[str],
    n_folds: int = 5,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-fold target encoding for high-cardinality categorical columns.
    Prevents leakage by encoding train with out-of-fold statistics.
    Encodes test with full-train statistics + smoothing.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()
    global_mean = float(y_train.mean())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for col in high_card_cols:
        if col not in X_train.columns:
            continue
        train_enc = np.full(len(X_train), global_mean, dtype=float)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            fold_map = (
                y_train.iloc[tr_idx]
                .groupby(X_train[col].iloc[tr_idx])
                .agg(["mean", "count"])
            )
            fold_map.columns = ["mean", "count"]
            fold_map["smoothed"] = (
                (fold_map["count"] * fold_map["mean"] + smoothing * global_mean)
                / (fold_map["count"] + smoothing)
            )
            train_enc[val_idx] = (
                X_train[col].iloc[val_idx]
                .map(fold_map["smoothed"])
                .fillna(global_mean)
                .values
            )
        X_train[col] = train_enc

        # Test: encode with full-train stats
        full_map = (
            y_train.groupby(X_train[col])
            .agg(["mean", "count"])
        )
        # At this point X_train[col] is already numeric (encoded above),
        # so we fall back to global_mean for test
        # Re-compute from original: store original values before encoding for test
        # (handled below via separate flow)
        X_test[col] = global_mean  # will be overwritten properly

    # Redo test encoding from scratch using original X_train (pre-encoding)
    return X_train, X_test


def _target_encode_proper(
    X_tr_orig: pd.DataFrame,
    y_train: pd.Series,
    X_te_orig: pd.DataFrame,
    high_card_cols: list[str],
    n_folds: int = 5,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proper cross-fold target encoding keeping original category values intact.
    """
    X_train = X_tr_orig.copy()
    X_test  = X_te_orig.copy()
    global_mean = float(y_train.mean())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for col in high_card_cols:
        if col not in X_train.columns:
            continue

        train_enc = np.full(len(X_train), global_mean, dtype=float)
        orig_train_col = X_tr_orig[col].astype(str)

        for tr_idx, val_idx in kf.split(X_train):
            stats = pd.DataFrame({
                "cat": orig_train_col.iloc[tr_idx].values,
                "y":   y_train.iloc[tr_idx].values,
            }).groupby("cat")["y"].agg(["mean", "count"])
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            train_enc[val_idx] = (
                orig_train_col.iloc[val_idx]
                .map(stats["smoothed"])
                .fillna(global_mean)
                .values
            )
        X_train[col] = train_enc

        # Full-train map for test
        full_stats = pd.DataFrame({
            "cat": orig_train_col.values,
            "y":   y_train.values,
        }).groupby("cat")["y"].agg(["mean", "count"])
        full_stats["smoothed"] = (
            (full_stats["count"] * full_stats["mean"] + smoothing * global_mean)
            / (full_stats["count"] + smoothing)
        )
        X_test[col] = (
            X_te_orig[col].astype(str)
            .map(full_stats["smoothed"])
            .fillna(global_mean)
            .values
        )

    return X_train, X_test


def _drop_id_cols(df: pd.DataFrame,
                  id_cols: list[str] | None = None) -> pd.DataFrame:
    default_ids = ["name", "_id", "host_name"]
    to_drop = [c for c in (id_cols or default_ids) if c in df.columns]
    return df.drop(columns=to_drop)


def _prepare_X(df: pd.DataFrame,
               target_col: str,
               drop_cols: list[str] | None = None) -> pd.DataFrame:
    drop = list(drop_cols or [])
    if target_col in df.columns:
        drop.append(target_col)
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = _extract_date_features(X)
    X = _drop_id_cols(X)
    X = _add_missing_indicators(X)
    return X


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    # All remaining categoricals get OrdinalEncoder (target encoding done separately)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.set_output(transform="pandas")
    return ct


def _build_model(model_name: str):
    """Return a regressor instance for the given model name."""
    if model_name == "ridge":
        return Ridge(alpha=10.0)
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_features=0.5,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            min_samples_leaf=10, random_state=42
        )
    if model_name == "lightgbm":
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is not installed")
        return lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    if model_name == "xgboost":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=1000, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    if model_name == "catboost":
        if not _CB_AVAILABLE:
            raise ImportError("catboost is not installed")
        return CatBoostRegressor(
            random_seed=42, 
            thread_count=-1,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _should_log_target(y: pd.Series) -> bool:
    """Return True if applying log1p to the target is likely beneficial."""
    return float(y.skew()) > 1.0 and float(y.min()) >= 0


# ── Full pipeline builder ────────────────────────────────────────────────────

def _build_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, bool, list[str]]:
    """
    Apply target encoding to high-cardinality cols, return processed
    X_train / X_test and (possibly log-transformed) y_train.
    Returns: X_train_enc, X_test_enc, y_enc, log_transformed, high_card_cols
    """
    # Target encoding for high-cardinality categoricals
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    high_card = [c for c in cat_cols if X_train[c].nunique() > 10]

    if high_card:
        X_train, X_test = _target_encode_proper(X_train, y_train, X_test, high_card)

    # Log-transform target
    log_transformed = _should_log_target(y_train)
    y_enc = np.log1p(y_train) if log_transformed else y_train.copy()

    return X_train, X_test, y_enc, log_transformed, high_card


# ── Main tools class ──────────────────────────────────────────────────────────

class MLTools:
    """Regression pipeline utilities exposed as agent tools."""

    AVAILABLE_MODELS: list[str] = ["ridge", "random_forest", "gradient_boosting"] + (
        ["lightgbm"] if _LGB_AVAILABLE else []
    ) + (
        ["xgboost"] if _XGB_AVAILABLE else []
    ) + (
        ["catboost"] if _CB_AVAILABLE else []
    )

    @staticmethod
    def prepare_features(path: str, target_col: str,
                          drop_cols: list[str] | None = None) -> dict[str, Any]:
        """Summarise feature types after preprocessing."""
        df = pd.read_csv(path)
        X = _prepare_X(df, target_col, drop_cols)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        high_card = [c for c in cat_cols if X[c].nunique() > 10]
        log_target = _should_log_target(df[target_col]) if target_col in df.columns else False
        return {
            "feature_columns": list(X.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "high_cardinality_columns": high_card,
            "n_samples": len(df),
            "n_features": len(X.columns),
            "recommend_log_target": log_target,
            "available_models": MLTools.AVAILABLE_MODELS,
        }

    @staticmethod
    def train_and_evaluate(path: str, target_col: str,
                            model_name: str = "lightgbm",
                            drop_cols: list[str] | None = None,
                            test_size: float = 0.2,
                            cv_folds: int = 5) -> dict[str, Any]:
        """
        Train a regressor and return CV + hold-out metrics (MSE, RMSE, MAE, R2).
        Automatically applies:
        - target encoding for high-cardinality categoricals
        - log1p target transform when skewness > 1
        - missing value indicators
        """
        df = pd.read_csv(path)
        X_full = _prepare_X(df, target_col, drop_cols)
        y_full = df[target_col]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, test_size=test_size, random_state=42
        )

        X_tr_enc, X_te_enc, y_tr_enc, log_transformed, _ = _build_full_pipeline(
            X_tr, y_tr, X_te, model_name
        )

        preprocessor = _build_preprocessor(X_tr_enc)
        reg = _build_model(model_name)
        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])
        pipeline.fit(X_tr_enc, y_tr_enc)

        raw_pred = pipeline.predict(X_te_enc)
        y_pred = np.expm1(raw_pred) if log_transformed else raw_pred
        y_pred = np.clip(y_pred, 0, 365)

        holdout = {
            "mse":  round(float(mean_squared_error(y_te, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
            "mae":  round(float(mean_absolute_error(y_te, y_pred)), 4),
            "r2":   round(float(r2_score(y_te, y_pred)), 4),
        }

        # CV on full data (encode within each fold to prevent leakage)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_mse_scores = []
        for tr_idx, val_idx in kf.split(X_full):
            X_cv_tr = X_full.iloc[tr_idx].copy()
            X_cv_val = X_full.iloc[val_idx].copy()
            y_cv_tr = y_full.iloc[tr_idx]
            y_cv_val = y_full.iloc[val_idx]

            X_cv_tr_enc, X_cv_val_enc, y_cv_enc, log_cv, _ = _build_full_pipeline(
                X_cv_tr, y_cv_tr, X_cv_val, model_name
            )
            pre = _build_preprocessor(X_cv_tr_enc)
            m = _build_model(model_name)
            pl = Pipeline([("pre", pre), ("reg", m)])
            pl.fit(X_cv_tr_enc, y_cv_enc)

            raw_cv = pl.predict(X_cv_val_enc)
            cv_pred = np.expm1(raw_cv) if log_cv else raw_cv
            cv_pred = np.clip(cv_pred, 0, 365)
            cv_mse_scores.append(mean_squared_error(y_cv_val, cv_pred))

        cv_arr = np.array(cv_mse_scores)

        # Save trained pipeline
        model_path = MODELS_DIR / f"{model_name}.pkl"
        pipeline.fit(X_tr_enc, y_tr_enc)   # re-fit on tr split for saving
        with open(model_path, "wb") as f:
            pickle.dump({
                "pipeline": pipeline,
                "log_transformed": log_transformed,
            }, f)

        return {
            "model": model_name,
            "holdout_metrics": holdout,
            "cv_mse_mean":  round(float(cv_arr.mean()), 4),
            "cv_mse_std":   round(float(cv_arr.std()), 4),
            "cv_rmse_mean": round(float(np.sqrt(cv_arr.mean())), 4),
            "log_transformed": log_transformed,
            "model_path": str(model_path),
        }

    @staticmethod
    def compare_models(path: str, target_col: str,
                        drop_cols: list[str] | None = None) -> dict[str, Any]:
        """Train and compare all available regressors; rank by CV MSE (lower = better)."""
        results = {}
        for m in MLTools.AVAILABLE_MODELS:
            try:
                res = MLTools.train_and_evaluate(
                    path, target_col, model_name=m, drop_cols=drop_cols
                )
                results[m] = {
                    "cv_mse":  res["cv_mse_mean"],
                    "cv_std":  res["cv_mse_std"],
                    "cv_rmse": res["cv_rmse_mean"],
                    "holdout": res["holdout_metrics"],
                    "log_transformed": res.get("log_transformed", False),
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
                            top_n: int = 20) -> dict[str, Any]:
        """Feature importances via LightGBM (or RF as fallback)."""
        df = pd.read_csv(path)
        X = _prepare_X(df, target_col, drop_cols)
        y = df[target_col]

        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        high_card = [c for c in cat_cols if X[c].nunique() > 10]
        X_enc, _, y_enc, log_t, _ = _build_full_pipeline(X, y, X.copy(), "lightgbm" if _LGB_AVAILABLE else "random_forest")

        preprocessor = _build_preprocessor(X_enc)
        if _LGB_AVAILABLE:
            reg = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1, n_jobs=-1)
        else:
            reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])
        pipeline.fit(X_enc, y_enc)

        importances = pipeline.named_steps["reg"].feature_importances_
        feature_names = pipeline.named_steps["pre"].get_feature_names_out()
        # Fallback if get_feature_names_out not available
        try:
            feature_names = list(feature_names)
        except Exception:
            feature_names = [f"f{i}" for i in range(len(importances))]

        imp_df = (
            pd.DataFrame({"feature": feature_names[:len(importances)],
                          "importance": importances[:len(feature_names)]})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        return {"feature_importances": imp_df.to_dict(orient="records")}

    @staticmethod
    def generate_submission(train_path: str, test_path: str,
                             target_col: str, model_name: str = "lightgbm",
                             drop_cols: list[str] | None = None,
                             output_path: str = "submission.csv") -> dict[str, Any]:
        """Train on the full training set and predict the test set. Saves submission CSV."""
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        X_train = _prepare_X(train_df, target_col, drop_cols)
        y_train = train_df[target_col]
        X_test  = _prepare_X(test_df, target_col, drop_cols)

        X_tr_enc, X_te_enc, y_enc, log_t, _ = _build_full_pipeline(
            X_train, y_train, X_test, model_name
        )
        preprocessor = _build_preprocessor(X_tr_enc)
        reg = _build_model(model_name)
        pipeline = Pipeline([("pre", preprocessor), ("reg", reg)])
        pipeline.fit(X_tr_enc, y_enc)

        raw = pipeline.predict(X_te_enc)
        preds = np.expm1(raw) if log_t else raw

        model_path = MODELS_DIR / f"{model_name}_full.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"pipeline": pipeline, "log_transformed": log_t}, f)

        preds = np.clip(preds, 0, 365)

        # Determine submission ID column
        id_col = None
        for c in ["id", "Id", "ID", "index"]:
            if c in test_df.columns:
                id_col = c
                break

        if id_col:
            submission = pd.DataFrame({id_col: test_df[id_col], "prediction": preds})
        else:
            submission = pd.DataFrame({"index": range(len(preds)), "prediction": preds})

        submission.to_csv(output_path, index=False)

        return {
            "submission_path": output_path,
            "n_predictions": len(preds),
            "pred_mean":  round(float(preds.mean()), 4),
            "pred_std":   round(float(preds.std()), 4),
            "pred_min":   round(float(preds.min()), 4),
            "pred_max":   round(float(preds.max()), 4),
        }

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        available = MLTools.AVAILABLE_MODELS
        return [
            {
                "name": "prepare_features",
                "description": (
                    "Summarise feature types after date extraction, missing indicators, "
                    "and ID dropping. Also reports high-cardinality columns and whether "
                    "log-transform of target is recommended."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the CSV training dataset (not a .pkl model file)"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "train_and_evaluate",
                "description": (
                    "Train a regressor with 5-fold CV and return MSE/RMSE/MAE/R2. "
                    "Automatically applies target encoding, log-target transform, "
                    "and missing indicators."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the CSV training dataset (not a .pkl model file)"},
                        "target_col": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "enum": available,
                            "default": "lightgbm" if _LGB_AVAILABLE else "gradient_boosting",
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
                "description": f"Compare all available regressors ({', '.join(available)}) by CV MSE.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the CSV training dataset (not a .pkl model file)"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "feature_importance",
                "description": "Compute feature importances using LightGBM (or RF as fallback).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the CSV training dataset (not a .pkl model file)"},
                        "target_col": {"type": "string"},
                        "drop_cols": {"type": "array", "items": {"type": "string"}, "default": []},
                        "top_n": {"type": "integer", "default": 20},
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
                        "train_path":    {"type": "string"},
                        "test_path":     {"type": "string"},
                        "target_col":    {"type": "string"},
                        "model_name":    {"type": "string", "default": "lightgbm"},
                        "drop_cols":     {"type": "array", "items": {"type": "string"}, "default": []},
                        "output_path":   {"type": "string", "default": "submission.csv"},
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
