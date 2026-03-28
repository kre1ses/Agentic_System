"""
Explorer agent — performs EDA using the EDATools suite.
"""
import json

from agents.base_agent import BaseAgent
from config import MODELS
from tools.eda_tools import EDATools


class ExplorerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(model=MODELS["explorer"], **kwargs)
        self.name = "Explorer"
        self.role = (
            "You are an expert data analyst. "
            "Use the available tools to explore the dataset thoroughly. "
            "Call every EDA tool in sequence, then write a concise Markdown report "
            "covering: dataset shape, missing values, class balance, feature types, "
            "correlations, outliers, and engineering recommendations. "
            "Be specific and data-driven."
        )
        self._eda = EDATools()
        self._dispatchers = [self._eda]
        self.tools = EDATools.get_tool_definitions()

    def explore(self, dataset_path: str, target_col: str) -> dict:
        """
        Run full EDA and return a structured report dict + narrative.
        """
        prompt = (
            f"Dataset path: {dataset_path}\n"
            f"Target column: {target_col}\n\n"
            "Please use all available EDA tools on this dataset. "
            "After calling the tools, write a clear summary report."
        )
        narrative = self.run(
            prompt,
            rag_query="regression EDA target distribution skewness missing values correlation outliers rental occupancy datetime features",
        )

        # Also run the full MCP EDA for a structured report
        from tools.mcp_interface import MCPInterface
        mcp = MCPInterface()
        structured = mcp.run_full_eda(dataset_path, target_col)
        structured["narrative"] = narrative

        self.store.log_eda(structured, agent=self.name)
        return structured
