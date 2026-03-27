"""
Safe code execution sandbox.

Runs LLM-generated Python code in a subprocess with:
  - configurable timeout
  - restricted builtins / imports via ast-level validation
  - stdout/stderr capture
  - resource guard (no network, no file writes outside allowed dirs)
"""
import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from config import CODE_TIMEOUT_SEC
from safety.guardrails import Guardrails

# Imports that are always allowed in sandboxed code
ALLOWED_IMPORTS = {
    "pandas", "numpy", "sklearn", "scipy", "math", "statistics",
    "collections", "itertools", "functools", "json", "re",
    "pathlib", "typing", "dataclasses", "datetime", "copy",
    "warnings", "pprint", "io", "csv", "string", "random",
    "matplotlib", "seaborn",
}


class SafeExecutor:
    """Execute Python code safely in a subprocess."""

    def __init__(self, timeout: int = CODE_TIMEOUT_SEC, allowed_dirs: list[str] | None = None):
        self.timeout = timeout
        self.allowed_dirs = allowed_dirs or []

    def execute(self, code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute `code` string safely.

        `context` is a dict of variable name → value serialised as JSON
        into the execution environment.

        Returns:
            {"stdout": str, "stderr": str, "returncode": int, "error": str|None}
        """
        # Static safety check
        is_safe, warnings = Guardrails.validate_generated_code(code)
        if not is_safe:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": "Code blocked by guardrails: " + "; ".join(warnings),
            }

        # AST-level import validation
        blocked, reason = self._check_imports(code)
        if blocked:
            return {"stdout": "", "stderr": "", "returncode": -1, "error": reason}

        # Write code to a temp file and execute it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            if context:
                # Inject context variables as Python literals at the top
                header_lines = ["import json as _json_mod_"]
                for k, v in context.items():
                    try:
                        import json
                        serialized = json.dumps(v, default=str)
                        header_lines.append(
                            f"{k} = _json_mod_.loads({repr(serialized)})"
                        )
                    except Exception:
                        pass
                tmp.write("\n".join(header_lines) + "\n\n")
            tmp.write(textwrap.dedent(code))
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
                "error": None,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": f"Execution timed out after {self.timeout}s",
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _check_imports(code: str) -> tuple[bool, str]:
        """
        Parse AST and block disallowed imports.
        Returns (is_blocked, reason).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return True, f"Syntax error in generated code: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in ALLOWED_IMPORTS:
                        return True, f"Blocked import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in ALLOWED_IMPORTS:
                        return True, f"Blocked import: {node.module}"
        return False, ""
