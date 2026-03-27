"""
Multi-Agent Binary Classification System
=========================================

Entry point for the full Planner-Explorer-Engineer-Builder-Critic pipeline.

Usage:
    python main.py                              # uses built-in sample dataset
    python main.py --dataset path/to/data.csv --target churn
    python main.py --no-llm                     # run in rule-based mode (no API key needed)
    python main.py --report-only <run_id>       # print report for existing run
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import ANTHROPIC_API_KEY, TARGET_COL, TEST_PATH, TRAIN_PATH
from evaluation.agent_eval import AgentEvaluator
from evaluation.metrics import ModelMetrics
from memory.experiment_store import ExperimentStore
from rag.knowledge_base import KnowledgeBase


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Regression System")
    parser.add_argument("--dataset", default=TRAIN_PATH,
                        help="Path to train CSV (default: data/train.csv)")
    parser.add_argument("--test",    default=TEST_PATH,
                        help="Path to test CSV for submission generation")
    parser.add_argument("--target",  default=TARGET_COL,
                        help="Name of the target column")
    parser.add_argument("--submission", default="submission.csv",
                        help="Output path for Kaggle submission file")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls (rule-based fallbacks)")
    parser.add_argument("--report-only", metavar="RUN_ID",
                        help="Print benchmark report for an existing run and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent logs")
    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 60)
    print("  Intelligent Multi-Agent System")
    print("  Regression | Rental Occupancy Prediction (MSE)")
    print("=" * 60)


def report_only(run_id: str):
    store = ExperimentStore(run_id=run_id)
    evaluator = AgentEvaluator(store)
    report = evaluator.evaluate()
    print(AgentEvaluator.format_report(report))


def run_pipeline(dataset_path: str, target_col: str,
                 test_path: str | None = None,
                 submission_path: str = "submission.csv",
                 verbose: bool = True) -> dict:
    """
    Orchestrate the full pipeline.
    Returns the final results dict.
    """
    from agents.coordinator import CoordinatorAgent

    # Shared infrastructure
    kb = KnowledgeBase()
    store = ExperimentStore()

    print(f"\n  Knowledge base: {len(kb)} chunks indexed")
    print(f"  Experiment store: run_id={store.run_id}")

    if not ANTHROPIC_API_KEY:
        print("\n  [!] ANTHROPIC_API_KEY not set.")
        print("      Running in rule-based (no-LLM) mode.")
        print("      Set the env var for full agent reasoning.\n")

    # Validate inputs
    from safety.guardrails import Guardrails
    safe, reason = Guardrails.validate_file_path(dataset_path)
    if not safe:
        print(f"  ✗ Invalid dataset path: {reason}")
        sys.exit(1)

    import pandas as pd
    cols = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    safe, reason = Guardrails.validate_column_name(target_col, cols)
    if not safe:
        print(f"  ✗ {reason}")
        sys.exit(1)

    # Run coordinator
    coordinator = CoordinatorAgent(kb=kb, store=store, verbose=verbose)
    result = coordinator.solve(dataset_path, target_col,
                               test_path=test_path,
                               submission_path=submission_path)

    # Extended evaluation
    print("\n" + "-" * 60)
    print("  Extended Evaluation")
    print("-" * 60)
    if result.get("model_path") and Path(result["model_path"]).exists():
        metrics = ModelMetrics.evaluate_saved_model(
            result["model_path"], dataset_path, target_col,
            drop_cols=result.get("feature_decisions", {}).get("drop_columns", []),
        )
        print(ModelMetrics.format_report(metrics))
    else:
        print("  (No saved model found for extended evaluation)")
    if result.get("submission", {}).get("submission_path"):
        print(f"\n  Kaggle submission: {result['submission']['submission_path']}")

    # Agent benchmark
    print("\n" + "-" * 60)
    evaluator = AgentEvaluator(store)
    bench = evaluator.evaluate()
    print(AgentEvaluator.format_report(bench))

    # Save final JSON report
    report_path = Path("experiments") / store.run_id / "final_report.json"
    with open(report_path, "w") as f:
        json.dump({**result, "agent_benchmark": bench}, f, indent=2, default=str)
    print(f"\n  Full report saved: {report_path}")

    return result


def main():
    print_banner()
    args = parse_args()

    if args.report_only:
        report_only(args.report_only)
        return

    if args.no_llm:
        # Override API key to force fallback mode
        os.environ["ANTHROPIC_API_KEY"] = ""
        import config
        config.ANTHROPIC_API_KEY = ""

    run_pipeline(
        dataset_path=args.dataset,
        target_col=args.target,
        test_path=args.test if Path(args.test).exists() else None,
        submission_path=args.submission,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
