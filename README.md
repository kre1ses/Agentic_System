# Multi-Agent  System

Intelligent multi-agent system for automated regression task on tabular (Kaggle-format) datasets.
Implements the full **Planner → Explorer → Engineer → Builder ↔ Critic** pipeline using the Anthropic Claude API.

---

## Architecture

```
main.py
  └── CoordinatorAgent          (orchestrates all phases)
        ├── PlannerAgent         (decomposes task into steps)
        ├── ExplorerAgent        (EDA via EDATools)
        │     └── [Critic review]
        ├── EngineerAgent        (feature decisions)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        ├── BuilderAgent         (model training & comparison)
        │     └── [Critic loop, up to MAX_CRITIQUE_ROUNDS]
        └── CriticAgent          (reviews every phase output)

tools/
  ├── EDATools        — load_dataset, basic_statistics, missing_values_report,
  │                     class_balance, correlation_analysis, outlier_detection,
  │                     feature_types_recommendation
  ├── MLTools         — prepare_features, train_and_evaluate, compare_models,
  │                     feature_importance
  └── MCPInterface    — MCP-style JSON-RPC facade over EDA + ML tools

rag/
  └── KnowledgeBase   — TF-IDF retrieval over 20 Kaggle best-practice chunks

memory/
  └── ExperimentStore — JSON-file persistence of every event across runs

safety/
  ├── Guardrails      — prompt-injection detection, path/column/model validation
  └── SafeExecutor    — AST-checked subprocess sandbox for generated code

evaluation/
  ├── ModelMetrics    — accuracy, F1, ROC-AUC, PR-AUC, Brier score
  └── AgentEvaluator  — critique convergence, AUC trajectory, agent participation
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

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (optional — system runs in rule-based mode without it)
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run on the built-in sample dataset
python main.py

# 4. Run on your own dataset
python main.py --dataset path/to/data.csv --target your_target_column

# 5. Run without LLM calls (fast rule-based mode for testing)
python main.py --no-llm

# 6. Print benchmark report for a past run
python main.py --report-only 20260327_122749
```

---

## Dataset

The built-in sample is a synthetic **customer churn** dataset (`data/customer_churn.csv`):
- 2,000 rows, 15 columns
- Mix of numeric + categorical features
- ~8–15% intentional missing values
- Highly correlated pair (`total_charges` / `total_charges_v2`)
- Skewed feature (`data_usage_gb`)
- High-cardinality ID column (`customer_id`)
- Binary target: `churn` (0 = retained, 1 = churned)

Regenerate with: `python data/generate_dataset.py`

---

## Evaluation Criteria Coverage

| Criterion (20% each) | Implementation |
|----------------------|----------------|
| **Architecture & Interaction** | 6-agent pipeline with ReAct/BDI/Planner-Critic patterns; MCP interface |
| **Automation & Safety** | Fully automated agent communication; Guardrails, SafeExecutor sandbox, prompt-injection detection |
| **Documentation & Transparency** | README, docstrings, per-run JSON event logs, `experiments/` directory |
| **Model Quality & Robustness** | 3-model comparison + StratifiedKFold CV; sklearn Pipeline prevents leakage |
| **Benchmarking & Deployment** | ModelMetrics (AUC/F1/PR-AUC/Brier) + AgentEvaluator; saved `.pkl` model |

---

## Output Structure

```
experiments/
  <run_id>/
    plan_<id>.json          — Planner output
    eda_<id>.json           — Explorer EDA report
    feature_decisions_*.json
    model_result_*.json
    critique_*.json
    agent_benchmark.json    — Agent effectiveness report
    final_report.json       — Full consolidated report
  models/
    random_forest.pkl       — Best trained model
  index.json                — Cross-run index

knowledge/
  corpus.json               — RAG knowledge chunks
  tfidf_index.pkl           — Fitted TF-IDF vectorizer
```

---

## Configuration

Edit `config.py` to change:
- `MODELS` — which Claude model each agent uses
- `MAX_CRITIQUE_ROUNDS` — how many Critic→Executor loops to allow
- `CODE_TIMEOUT_SEC` — sandbox execution timeout
- `RAG_TOP_K` — number of knowledge chunks retrieved per query
