"""
Multi-Agent System
=========================================

Entry point for the full Planner-Explorer-Engineer-Builder-Critic pipeline.

Usage:
    python main.py --kaggle mws-ai-agents-2026 --submit   # full end-to-end
    python main.py --kaggle mws-ai-agents-2026            # download only
    python main.py                                        # pipeline only
    python main.py --submit                               # submit existing submission.csv
    python main.py --no-llm --submit --submit-message "baseline"
    python main.py --report-only <run_id>                 # print report for existing run
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import ACTIVE_LLM_PROVIDER, TARGET_COL, TEST_PATH, TRAIN_PATH
from evaluation.agent_eval import AgentEvaluator
from evaluation.metrics import ModelMetrics
from memory.experiment_store import ExperimentStore
from rag.knowledge_base import KnowledgeBase


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Regression System")
    parser.add_argument("--dataset", default=TRAIN_PATH,
                        help="Path to train CSV (default: data_2/train.csv)")
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
    parser.add_argument("--kaggle", metavar="COMPETITION",
                        help="Download Kaggle competition dataset by name and exit "
                             "(e.g. --kaggle mws-ai-agents-2026)")
    parser.add_argument("--kaggle-dir", default="./data_2",
                        help="Destination directory for Kaggle download (default: ./data_2)")
    parser.add_argument("--competition", default="mws-ai-agents-2026",
                        help="Kaggle competition name used for --submit "
                             "(default: mws-ai-agents-2026)")
    parser.add_argument("--submit", action="store_true",
                        help="Submit the submission CSV to Kaggle after the pipeline "
                             "(or standalone to submit an existing file)")
    parser.add_argument("--submit-message", default="",
                        help="Comment attached to the Kaggle submission "
                             "(default: auto-generated from run timestamp)")
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

    if ACTIVE_LLM_PROVIDER == "none":
        print("\n  [!] No LLM API key detected.")
        print("      Running in rule-based (no-LLM) mode.")
        print("      Set OPENROUTER_API_KEY / ANTHROPIC_API_KEY for full agent reasoning.\n")

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


def _ensure_kaggle_credentials() -> None:
    """
    Make sure Kaggle credentials are available.

    Priority order:
      1. ~/.kaggle/kaggle.json already exists — nothing to do.
      2. KAGGLE_USERNAME + KAGGLE_KEY env vars are set — nothing to do.
      3. Otherwise prompt for Kaggle username and API key separately,
         then save them to ~/.kaggle/kaggle.json for future use.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    print("\n  Kaggle credentials not found.")
    print("  Find your API key at: kaggle.com → Settings → API → Create New Token\n")

    import getpass

    username = input("  Kaggle username: ").strip()
    if not username:
        print("  [!] Username cannot be empty.")
        sys.exit(1)

    api_key = getpass.getpass("  Kaggle API key (hidden): ").strip()
    if not api_key:
        print("  [!] API key cannot be empty.")
        sys.exit(1)

    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": api_key}, indent=2))
    kaggle_json.chmod(0o600)
    print(f"  Credentials saved to {kaggle_json}")


def kaggle_download(competition: str, output_dir: str = "./data_2") -> None:
    """
    Download and extract a Kaggle competition dataset.

    Steps:
      1. Ensure Kaggle credentials exist (prompt if missing).
      2. Run: kaggle competitions download -c <competition> -p <output_dir>
      3. Extract the downloaded ZIP into <output_dir>.
         - Uses Python's built-in zipfile (cross-platform).
         - On Windows also shows the equivalent PowerShell command.
         - On Linux/macOS also shows the equivalent unzip command.
    """
    _ensure_kaggle_credentials()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading competition '{competition}' → {out.resolve()}")
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(out)]
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  [!] kaggle download failed (exit code {result.returncode}).")
        print("  Make sure the 'kaggle' package is installed and you have accepted")
        print(f"  the competition rules at https://www.kaggle.com/c/{competition}")
        sys.exit(result.returncode)

    zip_path = out / f"{competition}.zip"
    if not zip_path.exists():
        # kaggle sometimes names it differently — find any zip
        zips = list(out.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print(f"\n  [!] No ZIP file found in {out}. Nothing to extract.")
            return

    print(f"\n  Extracting {zip_path.name} → {out.resolve()}")

    is_windows = platform.system() == "Windows"
    if is_windows:
        print(f"  (PowerShell equivalent: "
              f"Expand-Archive -Path \"{zip_path}\" -DestinationPath \"{out}\")")
    else:
        print(f"  (Shell equivalent: unzip {zip_path} -d {out})")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)

    print(f"  Done. Files extracted to {out.resolve()}")
    extracted = [p.name for p in out.iterdir() if p != zip_path]
    if extracted:
        print("  Extracted: " + ", ".join(sorted(extracted)))


def kaggle_submit(competition: str, submission_path: str = "submission.csv",
                  message: str = "") -> None:
    """
    Submit a prediction file to a Kaggle competition.

    Runs: kaggle competitions submit -c <competition> -f <file> -m "<message>"

    Credentials are checked/prompted the same way as kaggle_download().
    """
    _ensure_kaggle_credentials()

    sub = Path(submission_path)
    if not sub.exists():
        print(f"\n  [!] Submission file not found: {sub.resolve()}")
        print("  Run the pipeline first to generate it:")
        print("    python main.py")
        sys.exit(1)

    if not message:
        from datetime import datetime
        message = f"auto-submit {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition,
        "-f", str(sub),
        "-m", message,
    ]
    print(f"\n  Submitting '{sub}' to competition '{competition}'")
    print(f"  Message: {message}")
    print(f"  Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n  [!] Submission failed (exit code {result.returncode}).")
        print("  Check that you have accepted the competition rules at "
              f"https://www.kaggle.com/c/{competition}")
        sys.exit(result.returncode)

    print("\n  Submission successful!")
    print(f"  View results: https://www.kaggle.com/c/{competition}/submissions")


def main():
    print_banner()
    args = parse_args()

    # Competition name: --kaggle value takes priority over --competition
    competition = args.kaggle or args.competition

    # ── Step 1: download ────────────────────────────────────────────────
    if args.kaggle:
        kaggle_download(args.kaggle, args.kaggle_dir)
        # Download-only: no pipeline, no submit requested → done
        if not args.submit and not args.report_only:
            return

    # ── Step 2: report-only shortcut ────────────────────────────────────
    if args.report_only:
        report_only(args.report_only)
        return

    # ── Step 3: standalone submit (no pipeline needed) ──────────────────
    # Triggered when --submit is set but there is no dataset to train on
    if args.submit and not Path(args.dataset).exists():
        kaggle_submit(competition, args.submission, args.submit_message)
        return

    # ── Step 4: validate dataset before pipeline ────────────────────────
    if not Path(args.dataset).exists():
        print(f"\n  [!] Dataset not found: {args.dataset}")
        print("  Download the competition data first:")
        print("    python main.py --kaggle mws-ai-agents-2026")
        print("  Or run the full end-to-end pipeline:")
        print("    python main.py --kaggle mws-ai-agents-2026 --submit")
        sys.exit(1)

    # ── Step 5: run pipeline ─────────────────────────────────────────────
    if args.no_llm:
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

    # ── Step 6: submit ───────────────────────────────────────────────────
    if args.submit:
        kaggle_submit(competition, args.submission, args.submit_message)


if __name__ == "__main__":
    main()
