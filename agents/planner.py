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
            "You are a senior ML architect. "
            "Your job is to decompose a regression task into a clear, "
            "ordered list of subtasks for a team of specialised agents: "
            "Explorer (EDA), Engineer (feature engineering), Builder (model training), "
            "and Critic (quality review). "
            "Always output a JSON object with key 'plan': a list of steps, "
            "each step having 'id', 'agent', 'action', and 'depends_on' fields. "
            "Be concise and practical."
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
            f"Task: REGRESSION (predict days of rental occupancy, evaluated by MSE)\n"
            f"EDA hints: {eda_hints or 'none yet'}\n\n"
            "Create an execution plan to build the best regression model minimising MSE. "
            "Include: EDA (target distribution, date features, missing values), "
            "feature engineering, model selection (Ridge/RF/GBM), "
            "cross-validation by MSE, critique loop, final training on full data, "
            "and submission file generation. "
            "Output ONLY valid JSON with key 'plan'."
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
