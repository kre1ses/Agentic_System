"""Central configuration for the multi-agent system."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Paths
DATA_DIR = BASE_DIR / "data_2"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MODELS_DIR = BASE_DIR / "experiments" / "models"

for d in [EXPERIMENTS_DIR, KNOWLEDGE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset (Kaggle competition: rental occupancy regression)
# Download with: python main.py --kaggle mws-ai-agents-2026
TRAIN_PATH = str(DATA_DIR / "train.csv")
TEST_PATH  = str(DATA_DIR / "test.csv")
SUBMISSION_TEMPLATE = str(DATA_DIR / "sample_submition.csv")
TARGET_COL = "target"

# Claude API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Model assignments per agent role ─────────────────────────────────────────
# Each provider section lists the recommended model per agent.
# Rationale: complex-reasoning agents (Planner, Critic, Coordinator) get the
# largest available model; high-throughput agents (Explorer, Engineer, Builder)
# get a smaller/faster model to reduce latency and cost.

MODELS_BY_PROVIDER = {
    # Anthropic Claude — best quality, paid
    "anthropic": {
        "validator":   "claude-haiku-4-5-20251001",
        "planner":     "claude-sonnet-4-6",
        "explorer":    "claude-haiku-4-5-20251001",
        "engineer":    "claude-haiku-4-5-20251001",
        "builder":     "claude-haiku-4-5-20251001",
        "critic":      "claude-sonnet-4-6",
        "coordinator": "claude-sonnet-4-6",
        "reporter":    "claude-sonnet-4-6",
    },
    # OpenRouter — free open-source tier (verified available 2026-03)
    # 70B for complex reasoning (Planner, Critic, Coordinator, Reporter)
    # 24B for high-throughput tool-calling agents (Validator, Explorer, Engineer, Builder)
    "openrouter": {
        "validator":   "mistralai/mistral-small-3.1-24b-instruct:free",
        "planner":     "meta-llama/llama-3.3-70b-instruct:free",
        "explorer":    "mistralai/mistral-small-3.1-24b-instruct:free",
        "engineer":    "mistralai/mistral-small-3.1-24b-instruct:free",
        "builder":     "mistralai/mistral-small-3.1-24b-instruct:free",
        "critic":      "meta-llama/llama-3.3-70b-instruct:free",
        "coordinator": "meta-llama/llama-3.3-70b-instruct:free",
        "reporter":    "meta-llama/llama-3.3-70b-instruct:free",
    },
    # VseGPT (youragents.me) — Russian proxy for many models
    "vsegpt": {
        "validator":   "mistralai/mistral-small-3.1-24b-instruct",
        "planner":     "openai/o3-mini",
        "explorer":    "mistralai/mistral-small-3.1-24b-instruct",
        "engineer":    "mistralai/mistral-small-3.1-24b-instruct",
        "builder":     "mistralai/mistral-small-3.1-24b-instruct",
        "critic":      "openai/o3-mini",
        "coordinator": "openai/o3-mini",
        "reporter":    "openai/o3-mini",
    },
    "huggingface": {
        "validator":   "meta-llama/Llama-3.1-8B-Instruct",
        "planner":     "Qwen/Qwen2.5-72B-Instruct",
        "explorer":    "meta-llama/Llama-3.1-8B-Instruct",
        "engineer":    "meta-llama/Llama-3.1-8B-Instruct",
        "builder":     "meta-llama/Llama-3.1-8B-Instruct",
        "critic":      "Qwen/Qwen2.5-72B-Instruct",
        "coordinator": "Qwen/Qwen2.5-72B-Instruct",
        "reporter":    "Qwen/Qwen2.5-72B-Instruct",
    },
    # Fallback / no-LLM mode
    "none": {role: "none" for role in
             ["validator", "planner", "explorer", "engineer",
              "builder", "critic", "coordinator"]},
}

import os as _os
_provider = _os.environ.get("LLM_PROVIDER", "").lower() or (
    "anthropic"   if _os.environ.get("ANTHROPIC_API_KEY")  else
    "openrouter"  if _os.environ.get("OPENROUTER_API_KEY") else
    "vsegpt"      if _os.environ.get("VSEGPT_API_KEY")     else
    "huggingface" if _os.environ.get("HF_TOKEN")           else
    "none"
)
MODELS = MODELS_BY_PROVIDER.get(_provider, MODELS_BY_PROVIDER["none"])
ACTIVE_LLM_PROVIDER = _provider

# Agent behaviour
MAX_CRITIQUE_ROUNDS = 2
MAX_TOOL_RETRIES = 3
CODE_TIMEOUT_SEC = 60
MAX_TOKENS = 4096

# RAG
RAG_TOP_K = 3

# Task type
TASK_TYPE = "regression"          # "regression" | "binary_classification"
EVAL_METRICS = ["mse", "rmse", "mae", "r2"]
