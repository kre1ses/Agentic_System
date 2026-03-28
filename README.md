# Multi-Agent System

Intelligent multi-agent system for automated regression on tabular (Kaggle-format) datasets.
Implements the full **Validator → Planner → Explorer → Engineer → Builder ↔ Critic → Reporter** pipeline using the HuggingFace API (and OpenRouter / VseGPT / Anthropic Claude as alternative backends).

Note!!! This agent system works only for regression task.
---

## Architecture

```
main.py
  └── CoordinatorAgent          (orchestrates all phases)
        ├── ValidatorAgent       (Phase 0: input validation, schema & leakage checks)
        ├── PlannerAgent         (decomposes task into steps)
        ├── ExplorerAgent        (EDA via EDATools)
        │     └── [Critic review]
        ├── EngineerAgent        (feature engineering decisions, uses validation context)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        ├── BuilderAgent         (model training & comparison)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        ├── CriticAgent          (reviews every phase output)
        └── ReporterAgent        (generates Markdown reports in report/)

agents/
  ├── base_agent.py    — ReAct tool-use loop, LLM call wrappers
  ├── coordinator.py   — BDI orchestrator
  ├── validator.py     — Input validation, schema checks, leakage detection (Phase 0)
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
  ├── metrics.py      — MSE, RMSE, MAE, R²
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
| Fail-Fast Gate | ValidationAgent stops pipeline on critical data errors |
| Contract-First | ValidationAgent passes typed schema to downstream agents |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2.1. Set your LLM API key

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

### 2.2. Set your Kaggle API token

Make sure, you have accepted rules of the competition before you run the code.
!!! Sometimes KAGGLE_API_TOKEN can expire due to very long evaluation, so, be ready to submit the result by yourself.

```bash
export KAGGLE_API_TOKEN=...
```

### 3. Full end-to-end run (one command)

```bash
# Download dataset → run pipeline → submit to Kaggle
python main.py --kaggle mws-ai-agents-2026 --submit

# Same, with a custom submission message
python main.py --kaggle mws-ai-agents-2026 --submit --submit-message "xgboost v1"

# Rule-based mode (no LLM key needed)
python main.py --kaggle mws-ai-agents-2026 --no-llm --submit
```

### 4. Step-by-step

```bash
# Step 1 — download dataset only
python main.py --kaggle mws-ai-agents-2026
# equivalent standalone script:
python authenticate.py mws-ai-agents-2026 ./data_2

# Step 2 — run pipeline only
python main.py

# Step 3 — submit existing submission.csv
python main.py --submit
python main.py --submit --submit-message "run v2"

# Other
python main.py --report-only "YOUR_EXPERIMENT_RUN_ID"
```

<details>
<summary>Equivalent platform commands for the ZIP extraction</summary>

```powershell
# PowerShell (Windows)
Expand-Archive -Path "data_2/mws-ai-agents-2026.zip" -DestinationPath "./data_2"
```
```bash
# Linux / macOS
unzip data_2/mws-ai-agents-2026.zip -d ./data_2
```
</details>

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
| Validator / Explorer / Engineer / Builder | claude-haiku-4-5 | mistral-small-24b |

---

## Output Structure

```
experiments/
  <run_id>/
    validation_report_*.json  — Validator output (schema, leakage, drop candidates)
    plan_<id>.json            — Planner output
    eda_<id>.json             — Explorer EDA report
    feature_decisions_*.json  — Engineer decisions
    model_result_*.json       — Builder results
    critique_*.json           — Critic feedback
    agent_benchmark.json      — Agent effectiveness report
    final_report.json         — Full consolidated report
  models/
    model_name.pkl         — Best trained model

report/
  models.md                   — LLM-generated model selection justification
  llm_rationale.md            — model choice rationale

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
- `TASK_TYPE` — `"regression"`
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
  --submit                Submit submission CSV to Kaggle (standalone or after pipeline)
  --competition NAME      Competition name for --submit (default: mws-ai-agents-2026)
  --submit-message TEXT   Comment for the Kaggle submission (default: auto-timestamp)
  --quiet                 Suppress agent logs
```
