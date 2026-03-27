"""EDA tools for the Explorer agent."""
import json
import numpy as np
import pandas as pd
from typing import Any


class EDATools:
    """Collection of exploratory data analysis functions exposed as agent tools."""

    @staticmethod
    def load_dataset(path: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "path": path,
        }

    @staticmethod
    def basic_statistics(path: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_stats = {}
        for col in num_cols:
            s = df[col]
            numeric_stats[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "50%": round(float(s.median()), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurt()), 4),
            }

        cat_stats = {}
        for col in cat_cols:
            vc = df[col].value_counts()
            cat_stats[col] = {
                "unique": int(df[col].nunique()),
                "top5": {str(k): int(v) for k, v in vc.head(5).items()},
            }

        return {
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "numeric_stats": numeric_stats,
            "categorical_stats": cat_stats,
        }

    @staticmethod
    def missing_values_report(path: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        total = len(df)
        missing = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                missing[col] = {"count": n, "pct": round(n / total * 100, 2)}
        return {
            "total_rows": total,
            "columns_with_missing": missing,
            "complete_rows": int(df.dropna().shape[0]),
        }

    @staticmethod
    def target_distribution(path: str, target_col: str) -> dict[str, Any]:
        """Describe the regression target distribution."""
        df = pd.read_csv(path)
        s = df[target_col]
        zero_pct = round(float((s == 0).sum() / len(s) * 100), 2)
        return {
            "target_column": target_col,
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "min":    float(s.min()),
            "max":    float(s.max()),
            "skew":   round(float(s.skew()), 4),
            "pct_zero": zero_pct,
            "percentiles": {
                "25%": round(float(s.quantile(0.25)), 2),
                "50%": round(float(s.quantile(0.50)), 2),
                "75%": round(float(s.quantile(0.75)), 2),
                "90%": round(float(s.quantile(0.90)), 2),
                "95%": round(float(s.quantile(0.95)), 2),
            },
        }

    # keep old name as alias for backward-compat
    @staticmethod
    def class_balance(path: str, target_col: str) -> dict[str, Any]:
        return EDATools.target_distribution(path, target_col)

    @staticmethod
    def correlation_analysis(path: str, target_col: str, top_n: int = 10) -> dict[str, Any]:
        df = pd.read_csv(path)
        num_df = df.select_dtypes(include=[np.number])
        if target_col not in num_df.columns:
            # encode target if categorical
            num_df[target_col] = pd.factorize(df[target_col])[0]
        corr = num_df.corr()
        target_corr = (
            corr[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
        )
        feature_corr = {}
        for feat in target_corr.index:
            feature_corr[feat] = round(float(corr[target_col][feat]), 4)

        # multicollinearity: pairs with |corr| > 0.9
        high_corr_pairs = []
        cols = [c for c in num_df.columns if c != target_col]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = abs(float(corr.loc[cols[i], cols[j]]))
                if v > 0.9:
                    high_corr_pairs.append(
                        {"col1": cols[i], "col2": cols[j], "corr": round(v, 4)}
                    )

        return {
            "top_features_by_target_correlation": feature_corr,
            "high_multicollinearity_pairs": high_corr_pairs,
        }

    @staticmethod
    def outlier_detection(path: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n > 0:
                outliers[col] = {
                    "count": n,
                    "pct": round(n / len(df) * 100, 2),
                    "lower_fence": round(float(lower), 4),
                    "upper_fence": round(float(upper), 4),
                }
        return {"outlier_columns": outliers, "total_rows": len(df)}

    @staticmethod
    def feature_types_recommendation(path: str, target_col: str) -> dict[str, Any]:
        df = pd.read_csv(path)
        recs = {}
        for col in df.columns:
            if col == target_col:
                continue
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            if dtype in ("object", "category"):
                if n_unique <= 20:
                    recs[col] = "one_hot_encode"
                else:
                    recs[col] = "label_encode_or_drop"
            elif dtype in ("int64", "float64"):
                if n_unique <= 10:
                    recs[col] = "treat_as_categorical_or_keep"
                else:
                    recs[col] = "scale_numeric"
        return {"recommendations": recs}

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return tool schemas for Claude function calling."""
        return [
            {
                "name": "load_dataset",
                "description": "Load a CSV dataset and return schema info.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Path to CSV file"}},
                    "required": ["path"],
                },
            },
            {
                "name": "basic_statistics",
                "description": "Compute descriptive statistics for all columns.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "missing_values_report",
                "description": "Report missing values per column.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "target_distribution",
                "description": "Describe regression target distribution: mean, std, skew, % zeros, percentiles.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "correlation_analysis",
                "description": "Compute feature-target correlations and detect multicollinearity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10},
                    },
                    "required": ["path", "target_col"],
                },
            },
            {
                "name": "outlier_detection",
                "description": "Detect outliers using the IQR method.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "feature_types_recommendation",
                "description": "Recommend encoding/scaling strategy per feature.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target_col": {"type": "string"},
                    },
                    "required": ["path", "target_col"],
                },
            },
        ]

    def dispatch(self, tool_name: str, tool_input: dict) -> Any:
        """Route a tool call by name."""
        fn = getattr(self, tool_name, None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**tool_input)
        except Exception as exc:
            return {"error": str(exc)}
