"""
Coordinator agent — orchestrates the full Planner-Executor-Critic pipeline.

Workflow:
  1. Planner creates execution plan
  2. Explorer runs EDA → Critic reviews
  3. Engineer designs features → Critic reviews (up to MAX_CRITIQUE_ROUNDS)
  4. Builder trains models → Critic reviews (up to MAX_CRITIQUE_ROUNDS)
  5. Final evaluation report
"""
import json
from typing import Any

from agents.base_agent import BaseAgent
from agents.builder import BuilderAgent
from agents.critic import CriticAgent
from agents.engineer import EngineerAgent
from agents.explorer import ExplorerAgent
from agents.planner import PlannerAgent
from agents.reporter import ReporterAgent
from agents.validator import ValidationAgent
from config import MAX_CRITIQUE_ROUNDS, MODELS
from memory.experiment_store import ExperimentStore
from rag.knowledge_base import KnowledgeBase


class CoordinatorAgent(BaseAgent):
    def __init__(self, kb: KnowledgeBase | None = None,
                 store: ExperimentStore | None = None,
                 verbose: bool = True):
        super().__init__(model=MODELS["coordinator"], kb=kb, store=store, verbose=verbose)
        self.name = "Coordinator"
        self.role = (
            "You are the orchestrator of a multi-agent ML system. "
            "Coordinate Explorer, Engineer, Builder, and Critic agents "
            "to build the best regression model (minimise MSE) "
            "for rental occupancy prediction and generate a Kaggle submission file."
        )
        # Shared KB and store for all sub-agents
        self._validator = ValidationAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._planner = PlannerAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._explorer = ExplorerAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._engineer = EngineerAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._builder = BuilderAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._critic = CriticAgent(kb=self.kb, store=self.store, verbose=verbose)
        self._reporter = ReporterAgent(kb=self.kb, store=self.store, verbose=verbose)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, dataset_path: str, target_col: str,
              test_path: str | None = None,
              submission_path: str = "submission.csv") -> dict[str, Any]:
        """
        Run the full pipeline and return a results summary.
        """
        print(f"\n{'='*60}")
        print(f"  Multi-Agent Regression Task System")
        print(f"  Dataset : {dataset_path}")
        print(f"  Target  : {target_col}")
        print(f"  Run ID  : {self.store.run_id}")
        print(f"{'='*60}\n")

        # ── Phase 0: Validation ──────────────────────────────────────
        self._phase("0. Data Validation")
        validation_report = self._validator.validate(
            dataset_path, target_col, test_path=test_path
        )

        # ── Phase 1: Planning ────────────────────────────────────────
        self._phase("1. Planning")
        plan = self._planner.create_plan(dataset_path, target_col)
        self._print_plan(plan)

        # ── Phase 2: EDA ─────────────────────────────────────────────
        self._phase("2. Exploratory Data Analysis")
        eda_report = self._explorer.explore(dataset_path, target_col)
        eda_critique = self._critic.review_eda(eda_report)
        self._print_critique("EDA", eda_critique)

        # ── Phase 3: Feature Engineering (with Critic loop) ──────────
        self._phase("3. Feature Engineering")
        feature_decisions = self._feature_loop(
            eda_report, dataset_path, target_col,
            validation_report=validation_report,
        )

        # ── Phase 4: Model Building (with Critic loop) ───────────────
        self._phase("4. Model Building")
        final_result = self._model_loop(
            dataset_path, target_col, feature_decisions
        )

        # ── Phase 5: Submission Generation ───────────────────────────
        submission_result = {}
        if test_path:
            self._phase("5. Generating Submission")
            from tools.ml_tools import MLTools
            use_ensemble = final_result.get("model") == "ensemble"
            submission_result = MLTools.generate_submission(
                train_path=dataset_path,
                test_path=test_path,
                target_col=target_col,
                model_name=final_result.get("model", "lightgbm"),
                drop_cols=feature_decisions.get("drop_columns", []),
                output_path=submission_path,
                use_ensemble=use_ensemble,
            )
            print(f"  Submission saved : {submission_result.get('submission_path')}")
            print(f"  Predictions      : n={submission_result.get('n_predictions')} | "
                  f"mean={submission_result.get('pred_mean')} | "
                  f"std={submission_result.get('pred_std')}")

        # ── Phase 6: LLM-generated Reports ──────────────────────────
        self._phase("6. Generating LLM Reports")
        report = self._compile_report(eda_report, feature_decisions,
                                       final_result, submission_result,
                                       validation_report)
        self._print_report(report)
        self._write_reports(dataset_path, target_col, report, eda_report)

        # ── Phase 7: Persist experiment results into RAG ─────────────
        self._phase("7. Updating Knowledge Base")
        try:
            kb_chunks = self.store.to_kb_chunks()
            self.kb.learn_from_experiment(kb_chunks)
            print(f"  Added {len(kb_chunks)} chunk(s) from this run to the knowledge base.")
            print(f"  Total KB size: {len(self.kb)} chunks.")
        except Exception as e:
            print(f"  [!] KB update failed (non-fatal): {e}")

        return report

    # ------------------------------------------------------------------
    # Critic loops
    # ------------------------------------------------------------------

    def _feature_loop(self, eda_report: dict, dataset_path: str,
                       target_col: str,
                       validation_report: dict | None = None) -> dict:
        decisions = {}
        feedback = ""
        for i in range(MAX_CRITIQUE_ROUNDS + 1):
            decisions = self._engineer.plan_features(
                eda_report, dataset_path, target_col,
                critic_feedback=feedback,
                validation_report=validation_report,
            )
            critique = self._critic.review_feature_decisions(decisions, eda_report)
            self._print_critique("Feature decisions", critique)
            if critique.get("approved") or critique.get("severity") == "ok":
                break
            if critique.get("severity") == "minor" and i >= 1:
                break
            feedback = "; ".join(
                critique.get("issues", []) + critique.get("suggestions", [])
            )
        return decisions

    def _model_loop(self, dataset_path: str, target_col: str,
                     feature_decisions: dict) -> dict:
        result = {}
        feedback = ""
        for i in range(MAX_CRITIQUE_ROUNDS + 1):
            result = self._builder.build(
                dataset_path, target_col, feature_decisions,
                critic_feedback=feedback,
            )
            critique = self._critic.review_model_results(result)
            self._print_critique("Model results", critique)
            if critique.get("approved") or critique.get("severity") == "ok":
                break
            if critique.get("severity") == "minor" and i >= 1:
                break
            feedback = "; ".join(
                critique.get("issues", []) + critique.get("suggestions", [])
            )
        return result

    # ------------------------------------------------------------------
    # Report compilation
    # ------------------------------------------------------------------

    def _compile_report(self, eda_report: dict, feature_decisions: dict,
                         model_result: dict,
                         submission_result: dict | None = None,
                         validation_report: dict | None = None) -> dict:
        metrics = model_result.get("holdout_metrics", {})
        return {
            "run_id": self.store.run_id,
            "dataset_shape": eda_report.get("dataset_info", {}).get("shape"),
            "target_distribution": eda_report.get("target_distribution", {}),
            "validation_report": validation_report or {},
            "feature_decisions": feature_decisions,
            "best_model": model_result.get("model"),
            "model_path": model_result.get("model_path"),
            "holdout_metrics": metrics,
            "cv_mse": model_result.get("cv_mse_mean"),
            "cv_rmse": model_result.get("cv_rmse_mean"),
            "submission": submission_result or {},
            "run_summary": self.store.get_run_summary(),
        }

    def _write_reports(self, dataset_path: str, target_col: str,
                        report: dict, eda_report: dict):
        """Call ReporterAgent to generate both Markdown reports via LLM."""
        from tools.ml_tools import MLTools
        try:
            comparison   = MLTools.compare_models(dataset_path, target_col)
            importances  = MLTools.feature_importance(dataset_path, target_col)
        except Exception as e:
            print(f"  [reporter] Could not fetch ML data: {e}")
            comparison  = report.get("holdout_metrics", {})
            importances = {}

        target_stats = eda_report.get("target_distribution", {})
        missing_stats = eda_report.get("missing_values", {})
        experiment_summary = report.get("run_summary", {})

        try:
            self._reporter.write_models_report(
                experiment_summary=experiment_summary,
                model_comparison=comparison,
                feature_importances=importances,
                target_stats=target_stats,
                missing_stats=missing_stats,
            )
        except Exception as e:
            print(f"  [reporter] models.md failed: {e}")

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def _phase(self, label: str):
        print(f"\n{'-'*60}")
        print(f"  {label}")
        print(f"{'-'*60}")

    def _print_plan(self, plan: dict):
        steps = plan.get("plan", [])
        for s in steps:
            print(f"  Step {s.get('id', '?')}: [{s.get('agent', '?')}] {s.get('action', '?')}")

    def _print_critique(self, phase: str, critique: dict):
        approved = critique.get("approved", True)
        severity = critique.get("severity", "ok")
        icon = "[ok]" if approved else "[x]"
        print(f"  {icon} Critique [{phase}]: severity={severity}")
        for issue in critique.get("issues", []):
            print(f"    [!] {issue}")
        for sug in critique.get("suggestions", []):
            print(f"    [+] {sug}")

    def _print_report(self, report: dict):
        print(f"\n  Best model   : {report.get('best_model')}")
        print(f"  CV MSE       : {report.get('cv_mse')}  (RMSE={report.get('cv_rmse')})")
        m = report.get("holdout_metrics", {})
        print(f"  Holdout      : MSE={m.get('mse')} | RMSE={m.get('rmse')} | "
              f"MAE={m.get('mae')} | R2={m.get('r2')}")
        sub = report.get("submission", {})
        if sub.get("submission_path"):
            print(f"  Submission   : {sub['submission_path']}")
        print(f"  Model saved  : {report.get('model_path')}")
        print(f"  Run ID       : {report.get('run_id')}")
