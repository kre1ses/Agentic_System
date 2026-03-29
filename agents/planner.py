"""
Planner agent.

Decomposes the regression task into an ordered list of subtasks
and returns a structured JSON execution plan.
"""
import json

from agents.base_agent import BaseAgent
from config import MODELS


class PlannerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["planner"], **kwargs)
        self.name = "Planner"
        self.role = (
            "You are a senior ML architect whose execution plan is literally run "
            "by the Coordinator — the order and selection of steps you output "
            "determines what happens. "
            "Available agents and what they do:\n"
            "  Explorer  — full EDA (distributions, missing values, correlations); "
            "always include unless a prior EDA result is already available.\n"
            "  Engineer  — feature engineering decisions (encoding, transforms, "
            "interaction features); Critic review is automatically embedded.\n"
            "  Builder   — model comparison, Optuna tuning, final training; "
            "Critic review is automatically embedded.\n"
            "  Critic    — you may add Critic steps as documentation, but they are "
            "executed automatically inside Engineer and Builder — do NOT rely on them "
            "for standalone review.\n"
            "Rules:\n"
            "  1. Use EXACTLY these agent names: Explorer, Engineer, Builder, Critic.\n"
            "  2. Omit a phase only when there is a clear reason (e.g. skip Explorer "
            "if the dataset is well-known and EDA is unnecessary).\n"
            "  3. You may repeat Engineer or Builder if the task warrants "
            "an extra iteration (e.g. Engineer twice for complex feature work).\n"
            "  4. Validation always runs before your plan — do NOT include it.\n"
            "  5. Submission generation and reporting run after your plan — do NOT include them.\n"
            "Output ONLY a JSON object with key 'plan' (list of steps) and optional "
            "key 'notes' (string with rationale)."
        )

    def create_plan(self, dataset_path: str, target_col: str,
                    eda_hints: str = "") -> dict:
        """
        Generate an execution plan for the given dataset.
        Returns a parsed plan dict.
        """
        prompt = (
            f"Dataset: {dataset_path}\n"
            f"Target column: {target_col}\n"
            f"Task: REGRESSION — predict days of rental occupancy (0-365), evaluated by MSE.\n"
            f"EDA hints: {eda_hints or 'none yet'}\n\n"
            "Create an execution plan. Your plan is the ACTUAL execution schedule — "
            "the Coordinator dispatches agents in the exact order you specify.\n\n"
            "Available agents: Explorer, Engineer, Builder (Critic is embedded automatically).\n"
            "Do NOT include Validation, Submission, or Reporting steps — those always run "
            "outside your plan.\n\n"
            "For each step include: 'id' (int), 'agent' (one of the names above), "
            "'action' (short description), 'depends_on' (list of step ids).\n\n"
            "Think about: does EDA add value here? Should feature engineering iterate? "
            "Is a second Builder pass useful after Critic feedback?\n\n"
            "Output ONLY valid JSON with keys 'plan' and optionally 'notes'."
        )
        raw = self.run(prompt, rag_query="regression pipeline planning MSE rental occupancy feature engineering gradient boosting")
        # Parse JSON from response
        plan = self._parse_plan(raw)
        self.store.log_plan(plan, agent=self.name)
        return plan

    @staticmethod
    def _parse_plan(raw: str) -> dict:
        # Try to extract JSON block
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            # Fallback plan if LLM output is malformed
            return {
                "plan": [
                    {"id": 1, "agent": "Explorer",  "action": "Run full EDA",                   "depends_on": []},
                    {"id": 2, "agent": "Engineer",  "action": "Feature engineering",             "depends_on": [1]},
                    {"id": 3, "agent": "Critic",    "action": "Review feature engineering",      "depends_on": [2]},
                    {"id": 4, "agent": "Engineer",  "action": "Apply Critic feedback",           "depends_on": [3]},
                    {"id": 5, "agent": "Builder",   "action": "Compare and train models",        "depends_on": [4]},
                    {"id": 6, "agent": "Critic",    "action": "Review model selection",          "depends_on": [5]},
                    {"id": 7, "agent": "Builder",   "action": "Retrain best model (if needed)",  "depends_on": [6]},
                    {"id": 8, "agent": "Builder",   "action": "Final evaluation and report",     "depends_on": [7]},
                ]
            }
