from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Sequence


AnalysisType = Literal["ast", "ruff"]
DEFAULT_ANALYSES: tuple[AnalysisType, ...] = ("ast", "ruff")


def _normalize_analyses(analyses: Sequence[str] | str | None) -> tuple[AnalysisType, ...]:
    if analyses is None:
        return DEFAULT_ANALYSES

    if isinstance(analyses, str):
        selected = [analyses]
    else:
        selected = list(analyses)

    if not selected:
        return DEFAULT_ANALYSES

    normalized: list[AnalysisType] = []
    for analysis in selected:
        choice = analysis.lower().strip()
        if choice not in DEFAULT_ANALYSES:
            raise ValueError(
                f"Unknown analysis type '{analysis}'. Use one or more of: ast, ruff."
            )
        if choice not in normalized:
            normalized.append(choice)
    return tuple(normalized)


def analyze_ast(code: str) -> list[str]:
    """Run AST-based checks on a Python source string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = "unknown location"
        if exc.lineno is not None:
            location = f"line {exc.lineno}"
            if exc.offset is not None:
                location += f", column {exc.offset}"
        message = exc.msg or "invalid syntax"
        return [f"AST: Syntax error at {location}: {message}"]

    findings: list[str] = ["AST: No syntax errors found."]

    has_compose = any(
        isinstance(node, ast.FunctionDef) and node.name == "compose"
        for node in tree.body
    )
    if not has_compose:
        findings.append("AST: Missing top-level function `compose`.")

    has_compose_call = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "compose"
        for node in tree.body
    )
    if not has_compose_call:
        findings.append("AST: Missing module-level call to `compose()`.")

    has_import = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
    if not has_import:
        findings.append("AST: No import statements found.")

    return findings


def analyze_ruff(code: str, *, filename_hint: str = "generated.py") -> list[str]:
    """Run Ruff against a Python source string."""
    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return ["Ruff: command not found. Install Ruff to enable Python linting."]

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / filename_hint
        code_path.write_text(code, encoding="utf-8")

        completed = subprocess.run(
            [ruff_bin, "check", str(code_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if not output:
        if completed.returncode == 0:
            return ["Ruff: No issues found."]
        return ["Ruff: Linting failed without output."]

    return [f"Ruff: {line}" for line in output.splitlines() if line.strip()]


def analyze(code: str, analyses: Sequence[str] | str | None = None) -> str:
    """Return a short report for the selected analyses.

    If `analyses` is empty or omitted, all available analyses run.
    """
    selected = _normalize_analyses(analyses)
    findings: list[str] = []

    if "ast" in selected:
        findings.extend(analyze_ast(code))

    if "ruff" in selected:
        findings.extend(analyze_ruff(code))

    return "\n".join(findings)