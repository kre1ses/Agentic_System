"""
Input validation and prompt-injection guardrails.

All untrusted strings (user input, dataset column names, file paths)
must pass through Guardrails before being embedded in agent prompts.
"""
import re
from pathlib import Path
from typing import Any


# Patterns that signal prompt-injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+prior",
    r"you\s+are\s+now",
    r"act\s+as\s+(a\s+)?(different|another|new)\s+(ai|assistant|model)",
    r"forget\s+everything",
    r"system\s*:\s*",
    r"<\s*/?system\s*>",
    r"<\s*/?prompt\s*>",
    r"\[INST\]",
    r"###\s*(system|instruction)",
]

# Dangerous Python imports/calls that must not appear in generated code
_DANGEROUS_CODE_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\(.*['\"]w['\"]",   # file writes
    r"\bsocket\b",
    r"\brequests\b",
    r"\burllib\b",
    r"\bshutil\.(rmtree|move|copy)",
    r"\bos\.(remove|unlink|rmdir|makedirs|system)",
    r"\bimport\s+ctypes",
    r"\bimport\s+subprocess",
]


class Guardrails:
    """Stateless validation utilities."""

    @staticmethod
    def check_prompt_injection(text: str) -> tuple[bool, str]:
        """
        Returns (is_safe, reason).
        is_safe=False means a potential injection was detected.
        """
        lower = text.lower()
        for pat in _INJECTION_PATTERNS:
            if re.search(pat, lower):
                return False, f"Potential prompt injection: matched pattern '{pat}'"
        return True, "ok"

    @staticmethod
    def validate_file_path(path: str, allowed_dirs: list[str] | None = None) -> tuple[bool, str]:
        """Ensure path is a real CSV file and within allowed directories."""
        p = Path(path)
        if not p.exists():
            return False, f"File does not exist: {path}"
        if not p.is_file():
            return False, f"Not a file: {path}"
        if p.suffix.lower() not in {".csv", ".tsv", ".parquet"}:
            return False, f"Unsupported file type: {p.suffix}"
        if allowed_dirs:
            resolved = p.resolve()
            if not any(str(resolved).startswith(str(Path(d).resolve())) for d in allowed_dirs):
                return False, f"File outside allowed directories: {path}"
        return True, "ok"

    @staticmethod
    def validate_column_name(col: str, df_columns: list[str]) -> tuple[bool, str]:
        """Check that a column name exists in the dataframe."""
        if col not in df_columns:
            return False, f"Column '{col}' not found. Available: {df_columns}"
        return True, "ok"

    @staticmethod
    def sanitize_string(text: str, max_length: int = 500) -> str:
        """Strip control characters and truncate."""
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        return text[:max_length]

    @staticmethod
    def validate_generated_code(code: str) -> tuple[bool, list[str]]:
        """
        Scan LLM-generated code for dangerous patterns.
        Returns (is_safe, [list of warnings]).
        """
        warnings = []
        for pat in _DANGEROUS_CODE_PATTERNS:
            if re.search(pat, code):
                warnings.append(f"Dangerous pattern found: {pat}")
        is_safe = len(warnings) == 0
        return is_safe, warnings

    @staticmethod
    def validate_model_name(name: str) -> tuple[bool, str]:
        allowed = {
            "ridge", "random_forest", "gradient_boosting",
            "lightgbm", "xgboost", "ensemble",
            # legacy names
            "logistic_regression",
        }
        if name not in allowed:
            return False, f"Model '{name}' not allowed. Choose from {allowed}"
        return True, "ok"

    @staticmethod
    def validate_agent_response(response: str) -> tuple[bool, str]:
        """Light check that an agent response doesn't contain role-confusion attempts."""
        safe, reason = Guardrails.check_prompt_injection(response)
        return safe, reason

    @staticmethod
    def validate_tool_input(tool_name: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate inputs to named tools.
        Currently enforces path safety and column existence where possible.
        """
        if "path" in tool_input:
            safe, reason = Guardrails.validate_file_path(tool_input["path"])
            if not safe:
                return False, reason
        if "model_name" in tool_input:
            safe, reason = Guardrails.validate_model_name(tool_input["model_name"])
            if not safe:
                return False, reason
        return True, "ok"
