# Multi-Agent System

Intelligent multi-agent system for automated regression on tabular (Kaggle-format) datasets.
Implements the full **Planner → Explorer → Engineer → Builder ↔ Critic → Reporter** pipeline using the Anthropic Claude API (and OpenRouter / VseGPT / HuggingFace as alternative backends).

---

## Architecture

```
main.py
  └── CoordinatorAgent          (orchestrates all phases)
        ├── PlannerAgent         (decomposes task into steps)
        ├── ExplorerAgent        (EDA via EDATools)
        │     └── [Critic review]
        ├── EngineerAgent        (feature engineering decisions)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        ├── BuilderAgent         (model training & comparison)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        ├── CriticAgent          (reviews every phase output)
        └── ReporterAgent        (generates Markdown reports in report/)

agents/
  ├── base_agent.py    — ReAct tool-use loop, LLM call wrappers
  ├── coordinator.py   — BDI orchestrator
  ├── planner.py       — Chain-of-Thought task decomposition
  ├── explorer.py      — EDA with EDATools
  ├── engineer.py      — Feature-engineering decision maker
  ├── builder.py       — Model training & selection
  ├── critic.py        — Structured critique & improvement suggestions
  └── reporter.py      — LLM-generated Markdown experiment reports

llm/
  ├── factory.py              — selects backend from env / config
  ├── types.py                — shared Message / Response types
  ├── anthropic_backend.py    — Anthropic Claude
  ├── openai_compat_backend.py — OpenRouter / VseGPT (OpenAI-compatible)
  └── hf_backend.py           — HuggingFace Serverless Inference

tools/
  ├── eda_tools.py     — load_dataset, basic_statistics, missing_values_report,
  │                      class_balance, correlation_analysis, outlier_detection,
  │                      feature_types_recommendation
  ├── ml_tools.py      — prepare_features, train_and_evaluate, compare_models,
  │                      feature_importance
  └── mcp_interface.py — MCP-style JSON-RPC facade over EDA + ML tools

rag/
  ├── knowledge_base.py   — TF-IDF retrieval over Kaggle best-practice chunks
  └── kaggle_knowledge.py — 20 curated knowledge chunks

memory/
  └── experiment_store.py — JSON-file persistence of every event across runs

safety/
  ├── guardrails.py   — prompt-injection detection, path/column/model validation
  └── sandbox.py      — AST-checked subprocess sandbox for generated code

evaluation/
  ├── metrics.py      — MSE, RMSE, MAE, R², PR-AUC, Brier score
  └── agent_eval.py   — critique convergence, metric trajectory, agent participation
```

### Agent Patterns Used
| Pattern | Where |
|---------|-------|
| ReAct (Reason + Act + Observe) | BaseAgent tool-use loop |
| Chain-of-Thought | Planner & Critic system prompts |
| BDI (Belief-Desire-Intention) | Coordinator phase management |
| Planner-Executor-Critic | `coordinator.py` outer loop |
| Supervisor-Worker | Coordinator → sub-agents |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the competition dataset (Kaggle)

```bash
# Download and extract into ./data_2/
python main.py --kaggle mws-ai-agents-2026

# Or use the standalone script
python authenticate.py mws-ai-agents-2026 ./data_2
```

On first run you will be prompted to paste your **Kaggle API token** (JSON) if
`~/.kaggle/kaggle.json` is not present. Get it from
**kaggle.com → Settings → API → Create New Token**.

Internally the script runs:
```bash
kaggle competitions download -c mws-ai-agents-2026 -p ./data_2
```
and extracts the ZIP using Python's built-in `zipfile` module (works on Windows,
Linux, and macOS). The equivalent platform commands are:

```powershell
# PowerShell (Windows)
Expand-Archive -Path "data_2/mws-ai-agents-2026.zip" -DestinationPath "./data_2"
```
```bash
# Linux / macOS
unzip data_2/mws-ai-agents-2026.zip -d ./data_2
```

### 3. Set your LLM API key

At least one backend is required for full agent reasoning. Without a key the
system runs in rule-based fallback mode.

```bash
# Anthropic Claude (best quality, paid)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter (free open-source models)
export OPENROUTER_API_KEY=sk-or-...

# VseGPT / youragents.me (Russian proxy)
export VSEGPT_API_KEY=...

# HuggingFace Serverless Inference (free)
export HF_TOKEN=hf_...
```

### 4. Run the pipeline

```bash
# Run on the competition dataset (default)
python main.py

# Rule-based mode (no API key needed, fast testing)
python main.py --no-llm

# Print benchmark report for a past run
python main.py --report-only "YOUR EXPERIMENT NAME"
```

---

## Dataset

Dataset is the **mws-ai-agents-2026** Kaggle competition (`data_2/`):
- `train.csv` — training set with target column
- `test.csv`  — test set for submission generation
- `sample_submition.csv` — submission format example
- `solution.csv` — reference solution (if available)

Download with:
```bash
python main.py --kaggle mws-ai-agents-2026
```

---

## LLM Backends

| Provider | Env var | Notes |
|----------|---------|-------|
| Anthropic | `ANTHROPIC_API_KEY` | Best quality (paid) |
| OpenRouter | `OPENROUTER_API_KEY` | Free open-source tier |
| VseGPT | `VSEGPT_API_KEY` | Russian proxy, many models |
| HuggingFace | `HF_TOKEN` | Free serverless inference |
| *(none)* | — | Rule-based fallback mode |

The active provider is auto-detected from whichever key is set.
Override with `LLM_PROVIDER=anthropic|openrouter|vsegpt|huggingface`.

Model assignments per agent (see `config.py`):

| Agent | Anthropic | OpenRouter |
|-------|-----------|------------|
| Planner / Critic / Coordinator / Reporter | claude-sonnet-4-6 | llama-3.3-70b |
| Explorer / Engineer / Builder | claude-haiku-4-5 | mistral-small-24b |

---

## Evaluation Criteria Coverage

| Criterion (20 % each) | Implementation |
|----------------------|----------------|
| **Architecture & Interaction** | 7-agent pipeline with ReAct/BDI/Planner-Critic patterns; MCP interface; pluggable LLM backends |
| **Automation & Safety** | Fully automated agent communication; Guardrails, SafeExecutor sandbox, prompt-injection detection |
| **Documentation & Transparency** | README, docstrings, per-run JSON event logs, `experiments/` directory, LLM-generated reports in `report/` |
| **Model Quality & Robustness** | 3-model comparison + StratifiedKFold CV; sklearn Pipeline prevents leakage |
| **Benchmarking & Deployment** | ModelMetrics (MSE/RMSE/MAE/R²) + AgentEvaluator; saved `.pkl` model; Kaggle submission CSV |

---

## Output Structure

```
experiments/
  <run_id>/
    plan_<id>.json            — Planner output
    eda_<id>.json             — Explorer EDA report
    feature_decisions_*.json  — Engineer decisions
    model_result_*.json       — Builder results
    critique_*.json           — Critic feedback
    agent_benchmark.json      — Agent effectiveness report
    final_report.json         — Full consolidated report
  models/
    random_forest.pkl         — Best trained model

report/
  models.md                   — LLM-generated model selection justification
  llm_rationale.md            — LLM-generated provider / model choice rationale

knowledge/
  corpus.json                 — RAG knowledge chunks
  tfidf_index.pkl             — Fitted TF-IDF vectorizer

submission.csv                — Kaggle submission file (generated by Builder)
```

---

## Configuration

Edit `config.py` to change:
- `MODELS_BY_PROVIDER` — which model each agent uses per provider
- `MAX_CRITIQUE_ROUNDS` — how many Critic→Executor loops to allow (default: 2)
- `CODE_TIMEOUT_SEC` — sandbox execution timeout (default: 60 s)
- `RAG_TOP_K` — number of knowledge chunks retrieved per query (default: 3)
- `TASK_TYPE` — `"regression"` or `"binary_classification"`
- `TRAIN_PATH` / `TEST_PATH` — default dataset paths

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --dataset PATH          Path to train CSV (default: data_2/train.csv)
  --test PATH             Path to test CSV for submission generation
  --target COL            Name of the target column (default: target)
  --submission PATH       Output path for Kaggle submission (default: submission.csv)
  --no-llm                Run in rule-based mode without LLM calls
  --report-only RUN_ID    Print benchmark report for an existing run and exit
  --kaggle COMPETITION    Download Kaggle competition dataset and exit
  --kaggle-dir DIR        Destination for Kaggle download (default: ./data_2)
  --quiet                 Suppress agent logs
```
