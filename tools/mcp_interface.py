"""
Model Context Protocol (MCP) style interface.

Provides a standardised JSON-RPC-like layer over EDATools and MLTools so that
any MCP-compatible client (or agent) can call tools through a single endpoint.
"""
import json
from typing import Any

from tools.eda_tools import EDATools
from tools.ml_tools import MLTools


class MCPInterface:
    """
    Thin MCP facade.

    Exposes `call(method, params)` where `method` is "<namespace>/<tool_name>"
    and `params` is a dict of keyword arguments.

    Namespaces:
        eda  – EDATools methods
        ml   – MLTools methods
    """

    def __init__(self):
        self._eda = EDATools()
        self._ml = MLTools()
        self._registry: dict[str, tuple[Any, dict]] = {}
        self._build_registry()

    def _build_registry(self):
        for defn in EDATools.get_tool_definitions():
            self._registry[f"eda/{defn['name']}"] = (self._eda, defn)
        for defn in MLTools.get_tool_definitions():
            self._registry[f"ml/{defn['name']}"] = (self._ml, defn)

    def list_tools(self) -> list[dict]:
        """Return all available tools with their schemas."""
        tools = []
        for method, (_, defn) in self._registry.items():
            tools.append({"method": method, "schema": defn})
        return tools

    def call(self, method: str, params: dict | None = None) -> dict[str, Any]:
        """
        Execute a tool call.

        Returns:
            {"result": ..., "error": None}  on success
            {"result": None, "error": "..."}  on failure
        """
        params = params or {}
        if method not in self._registry:
            return {"result": None, "error": f"Unknown method: {method}"}
        dispatcher, _ = self._registry[method]
        tool_name = method.split("/", 1)[1]
        result = dispatcher.dispatch(tool_name, params)
        if isinstance(result, dict) and "error" in result:
            return {"result": None, "error": result["error"]}
        return {"result": result, "error": None}

    def call_json(self, request_json: str) -> str:
        """Accept a JSON string, execute the call, return JSON string."""
        try:
            req = json.loads(request_json)
            method = req.get("method", "")
            params = req.get("params", {})
            resp = self.call(method, params)
        except Exception as e:
            resp = {"result": None, "error": str(e)}
        return json.dumps(resp, default=str)

    # ------------------------------------------------------------------
    # Convenience wrappers for common workflows
    # ------------------------------------------------------------------

    def run_full_eda(self, dataset_path: str, target_col: str) -> dict[str, Any]:
        """Run the complete EDA suite and return a combined report."""
        report = {}
        report["dataset_info"] = self.call("eda/load_dataset", {"path": dataset_path})["result"]
        report["statistics"] = self.call("eda/basic_statistics", {"path": dataset_path})["result"]
        report["missing"] = self.call("eda/missing_values_report", {"path": dataset_path})["result"]
        report["target_distribution"] = self.call(
            "eda/target_distribution", {"path": dataset_path, "target_col": target_col}
        )["result"]
        report["correlations"] = self.call(
            "eda/correlation_analysis",
            {"path": dataset_path, "target_col": target_col},
        )["result"]
        report["outliers"] = self.call("eda/outlier_detection", {"path": dataset_path})["result"]
        report["feature_recs"] = self.call(
            "eda/feature_types_recommendation",
            {"path": dataset_path, "target_col": target_col},
        )["result"]
        return report

    def run_model_comparison(self, dataset_path: str, target_col: str,
                              drop_cols: list[str] | None = None) -> dict[str, Any]:
        """Compare all models and return best model + metrics."""
        return self.call(
            "ml/compare_models",
            {"path": dataset_path, "target_col": target_col, "drop_cols": drop_cols or []},
        )["result"]
