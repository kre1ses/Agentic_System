"""Central configuration for the multi-agent system."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Paths
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MODELS_DIR = BASE_DIR / "experiments" / "models"

for d in [DATA_DIR, EXPERIMENTS_DIR, KNOWLEDGE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset (Kaggle competition: rental occupancy regression)
TRAIN_PATH = str(DATA_DIR / "train.csv")
TEST_PATH  = str(DATA_DIR / "test.csv")
SUBMISSION_TEMPLATE = str(DATA_DIR / "sample_submition.csv")
TARGET_COL = "target"

# Claude API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model assignments per agent role (balance quality vs cost)
MODELS = {
    "planner":     "claude-sonnet-4-6",
    "explorer":    "claude-haiku-4-5-20251001",
    "engineer":    "claude-haiku-4-5-20251001",
    "builder":     "claude-haiku-4-5-20251001",
    "critic":      "claude-sonnet-4-6",
    "coordinator": "claude-sonnet-4-6",
}

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
