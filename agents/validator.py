"""
Validation agent — input validation, schema checks, leakage detection.

Runs as Phase 0, before Planner and EDA.
Returns a structured validation_report and either allows the pipeline
to continue or raises SystemExit with a clear diagnostic message.
"""
import json
import sys

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from config import MODELS

_LEAKAGE_NAME_PATTERNS = [
    "target", "label", "outcome", "score", "result",
    "price_final", "final_price", "actual", "ground_truth",
]

_ID_NAME_PATTERNS = [
    "id", "_id", "idx", "index", "row_id", "rowid", "record_id", "key",
]

# Missing-value severity thresholds
_MISS_HIGH = 0.80
_MISS_WARN = 0.40

# Leakage correlation thresholds
_CORR_CRITICAL = 0.98
_CORR_HIGH = 0.90


class ValidationAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["validator"], **kwargs)
        self.name = "Validator"
        self.role = (
            "You are a data validation expert. "
            "Given the validation findings below, briefly summarise the key risks "
            "(leakage suspects, schema mismatches, target anomalies) in 3-5 sentences. "
            "Be concise and practical."
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        train_path: str,
        target_col: str,
        test_path: str | None = None,
    ) -> dict:
        """
        Run all validation checks and return a structured report.
        Stops the pipeline (sys.exit) if can_proceed=False.
        """
        report = self._run_checks(train_path, target_col, test_path)

        # Optional LLM summary for leakage suspects
        if report["leakage_suspects"] and MODELS.get("validator") != "none":
            try:
                summary = self.run(
                    f"Validation findings:\n{json.dumps(report, default=str)[:2000]}\n\n"
                    "Briefly summarise the leakage risk and any schema issues.",
                    rag_query="data leakage detection regression ID columns train test schema validation",
                )
                report["llm_summary"] = summary[:800]
            except Exception:
                pass

        self.store.log("validation_report", report, agent=self.name)

        if not report["can_proceed"]:
            print(f"\n  [Validator] PIPELINE STOPPED: {report['stop_reason']}")
            sys.exit(1)

        self._print_report(report)
        return report

    # ------------------------------------------------------------------
    # Core checks (pure pandas — no LLM required)
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        train_path: str,
        target_col: str,
        test_path: str | None,
    ) -> dict:
        risk_flags: list[dict] = []
        leakage_suspects: list[dict] = []
        recommended_actions: list[str] = []

        # ── 1. Load train ────────────────────────────────────────────
        try:
            train = pd.read_csv(train_path)
        except Exception as e:
            return self._fail(f"Cannot read train file: {e}")

        if train.empty:
            return self._fail("train.csv is empty")

        n_train, n_cols = train.shape

        # ── 2. Target checks ────────────────────────────────────────
        if target_col not in train.columns:
            return self._fail(
                f"Target column '{target_col}' not found in train. "
                f"Available: {list(train.columns[:10])}"
            )

        target_series = train[target_col]
        if not pd.api.types.is_numeric_dtype(target_series):
            try:
                target_series = pd.to_numeric(target_series, errors="raise")
            except Exception:
                return self._fail(
                    f"Target '{target_col}' is not numeric and cannot be coerced"
                )

        if target_series.isna().all():
            return self._fail("Target column is entirely NaN")

        target_null_pct = float(target_series.isna().mean())
        if target_null_pct > 0.5:
            return self._fail(f"Target column is {target_null_pct:.0%} NaN")

        if target_series.std() == 0:
            return self._fail("Target column is constant — regression is undefined")

        target_nunique = int(target_series.nunique())
        target_skew = float(target_series.skew())

        if target_nunique <= 10:
            risk_flags.append({
                "column": target_col,
                "risk": "low_cardinality_target",
                "severity": "medium",
                "detail": f"only {target_nunique} unique values — might be classification",
            })

        if abs(target_skew) > 2:
            risk_flags.append({
                "column": target_col,
                "risk": "high_skew_target",
                "severity": "medium",
                "detail": f"skewness={target_skew:.2f}",
            })
            recommended_actions.append("consider log-transform for target")

        # ── 3. Load and compare test ─────────────────────────────────
        n_test = None
        missing_test_cols: list[str] = []
        unexpected_test_cols: list[str] = []
        train_test_aligned = True

        if test_path:
            try:
                test = pd.read_csv(test_path)
            except Exception as e:
                risk_flags.append({
                    "column": "test",
                    "risk": "unreadable_test_file",
                    "severity": "high",
                    "detail": str(e),
                })
                test = None

            if test is not None:
                n_test = len(test)
                feature_cols = [c for c in train.columns if c != target_col]
                missing_test_cols = [c for c in feature_cols if c not in test.columns]
                unexpected_test_cols = [c for c in test.columns if c not in train.columns]

                if target_col in test.columns:
                    risk_flags.append({
                        "column": target_col,
                        "risk": "target_in_test",
                        "severity": "high",
                        "detail": "target column present in test — possible leakage",
                    })
                    leakage_suspects.append({
                        "column": target_col,
                        "risk": "target_in_test",
                        "severity": "high_leakage_risk",
                    })

                if missing_test_cols:
                    train_test_aligned = False
                    risk_flags.append({
                        "column": str(missing_test_cols[:5]),
                        "risk": "missing_test_columns",
                        "severity": "high",
                        "detail": f"{len(missing_test_cols)} feature(s) from train absent in test",
                    })
                    # Fatal if majority of features are missing
                    if len(missing_test_cols) > len(feature_cols) * 0.5:
                        return self._fail(
                            f"train/test critically misaligned: "
                            f"{len(missing_test_cols)}/{len(feature_cols)} columns missing in test"
                        )

        # ── 4. Feature type profiling ────────────────────────────────
        feature_type_map: dict[str, str] = {}
        for col in train.columns:
            if col == target_col:
                continue
            s = train[col]
            if pd.api.types.is_numeric_dtype(s):
                feature_type_map[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(s):
                feature_type_map[col] = "datetime"
            else:
                try:
                    sample = s.dropna().iloc[:50]
                    if len(sample) > 0:
                        pd.to_datetime(sample, infer_datetime_format=True)
                    feature_type_map[col] = "datetime-like"
                except Exception:
                    n_unique = s.nunique()
                    avg_len = s.dropna().astype(str).str.len().mean() if len(s.dropna()) > 0 else 0
                    if avg_len > 50:
                        feature_type_map[col] = "text-like"
                    elif n_unique > 0 and n_unique / max(len(s.dropna()), 1) > 0.9:
                        feature_type_map[col] = "high-cardinality-object"
                    else:
                        feature_type_map[col] = "categorical"

        # ── 5. ID-column detection ───────────────────────────────────
        drop_candidates: list[str] = []
        for col in train.columns:
            if col == target_col:
                continue
            col_lower = col.lower()
            is_id_name = any(
                col_lower == pat or col_lower.endswith(pat)
                for pat in _ID_NAME_PATTERNS
            )
            uniqueness_ratio = train[col].nunique() / n_train
            if is_id_name or uniqueness_ratio > 0.95:
                drop_candidates.append(col)
                risk_flags.append({
                    "column": col,
                    "risk": "id_like_column",
                    "severity": "medium",
                    "detail": f"uniqueness={uniqueness_ratio:.2f}",
                })
                recommended_actions.append(f"drop '{col}' from modeling features (ID-like)")

        # ── 6. Missing values profile ────────────────────────────────
        missingness_profile: dict[str, float] = {}
        for col in train.columns:
            if col == target_col:
                continue
            pct = float(train[col].isna().mean())
            if pct > 0:
                missingness_profile[col] = round(pct, 4)
            if pct >= _MISS_HIGH:
                risk_flags.append({
                    "column": col,
                    "risk": "high_missing_rate",
                    "severity": "high",
                    "detail": f"{pct:.0%} NaN",
                })
                if pct == 1.0:
                    drop_candidates.append(col)
                    recommended_actions.append(f"drop '{col}' (100% NaN)")
            elif pct >= _MISS_WARN:
                risk_flags.append({
                    "column": col,
                    "risk": "moderate_missing_rate",
                    "severity": "medium",
                    "detail": f"{pct:.0%} NaN",
                })

        # ── 7. Constant features ─────────────────────────────────────
        for col in train.columns:
            if col == target_col:
                continue
            if train[col].nunique(dropna=True) <= 1:
                if col not in drop_candidates:
                    drop_candidates.append(col)
                risk_flags.append({
                    "column": col,
                    "risk": "constant_column",
                    "severity": "medium",
                    "detail": "0 or 1 unique value",
                })
                recommended_actions.append(f"drop constant column '{col}'")

        # ── 8. Leakage heuristics ────────────────────────────────────
        numeric_features = [
            c for c in train.columns
            if c != target_col and pd.api.types.is_numeric_dtype(train[c])
        ]
        target_vals = target_series.fillna(target_series.median())

        for col in numeric_features:
            col_lower = col.lower()
            # Name-based check
            if any(pat in col_lower for pat in _LEAKAGE_NAME_PATTERNS):
                leakage_suspects.append({
                    "column": col,
                    "risk": "suspicious_name",
                    "severity": "warning",
                })
                continue
            # Correlation-based check
            try:
                col_vals = train[col].fillna(train[col].median())
                corr = abs(float(col_vals.corr(target_vals)))
                if corr > _CORR_CRITICAL:
                    leakage_suspects.append({
                        "column": col,
                        "risk": "near_perfect_correlation",
                        "severity": "high_leakage_risk",
                        "detail": f"corr={corr:.4f}",
                    })
                elif corr > _CORR_HIGH:
                    leakage_suspects.append({
                        "column": col,
                        "risk": "high_correlation",
                        "severity": "warning",
                        "detail": f"corr={corr:.4f}",
                    })
            except Exception:
                pass

        # ── 9. Duplicate rows ────────────────────────────────────────
        n_dup = int(train.duplicated().sum())
        if n_dup > 0:
            dup_pct = n_dup / n_train
            risk_flags.append({
                "column": "rows",
                "risk": "duplicate_rows",
                "severity": "medium" if dup_pct < 0.05 else "high",
                "detail": f"{n_dup} duplicate rows ({dup_pct:.1%})",
            })
            recommended_actions.append(f"deduplicate {n_dup} duplicate rows")

        # Deduplicate drop_candidates preserving order
        drop_candidates = list(dict.fromkeys(drop_candidates))

        return {
            "status": "pass",
            "can_proceed": True,
            "task_type": "regression",
            "target_column": target_col,
            "dataset_summary": {
                "n_rows_train": n_train,
                "n_rows_test": n_test,
                "n_features": n_cols - 1,
            },
            "schema_checks": {
                "train_test_aligned": train_test_aligned,
                "missing_test_columns": missing_test_cols,
                "unexpected_test_columns": unexpected_test_cols,
            },
            "target_stats": {
                "skewness": round(target_skew, 3),
                "n_unique": target_nunique,
                "null_pct": round(target_null_pct, 4),
            },
            "feature_type_map": feature_type_map,
            "drop_candidates": drop_candidates,
            "missingness_profile": missingness_profile,
            "leakage_suspects": leakage_suspects,
            "risk_flags": risk_flags,
            "recommended_actions": recommended_actions,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fail(reason: str) -> dict:
        return {
            "status": "fail",
            "can_proceed": False,
            "stop_reason": reason,
            "task_type": "regression",
            "target_column": "",
            "dataset_summary": {},
            "schema_checks": {"train_test_aligned": False,
                              "missing_test_columns": [], "unexpected_test_columns": []},
            "target_stats": {},
            "feature_type_map": {},
            "drop_candidates": [],
            "missingness_profile": {},
            "leakage_suspects": [],
            "risk_flags": [{"column": "pipeline", "risk": "fatal",
                            "severity": "critical", "detail": reason}],
            "recommended_actions": [],
        }

    def _print_report(self, report: dict) -> None:
        ds = report["dataset_summary"]
        sc = report["schema_checks"]
        print(f"  train={ds.get('n_rows_train')} rows | "
              f"test={ds.get('n_rows_test')} rows | "
              f"features={ds.get('n_features')}")
        print(f"  schema aligned: {sc.get('train_test_aligned')}")
        if report["drop_candidates"]:
            print(f"  drop candidates : {report['drop_candidates']}")
        suspects = [s["column"] for s in report["leakage_suspects"]]
        if suspects:
            print(f"  [!] leakage suspects: {suspects}")
        for flag in report["risk_flags"]:
            if flag["severity"] in ("high", "critical"):
                print(f"  [!] {flag['risk']} on '{flag['column']}': "
                      f"{flag.get('detail', '')}")
        for action in report["recommended_actions"][:6]:
            print(f"  [+] {action}")
