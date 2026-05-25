from __future__ import annotations

import ast


def analyze(code: str) -> str:
    """Return a short linter-style report for a Python code string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = "unknown location"
        if exc.lineno is not None:
            location = f"line {exc.lineno}"
            if exc.offset is not None:
                location += f", column {exc.offset}"
        message = exc.msg or "invalid syntax"
        return f"Syntax error at {location}: {message}"

    findings: list[str] = ["No syntax errors found."]

    has_compose = any(
        isinstance(node, ast.FunctionDef) and node.name == "compose"
        for node in tree.body
    )
    if not has_compose:
        findings.append("Missing top-level function `compose`.")

    has_compose_call = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "compose"
        for node in tree.body
    )
    if not has_compose_call:
        findings.append("Missing module-level call to `compose()`.")

    has_import = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
    if not has_import:
        findings.append("No import statements found.")

    return "\n".join(findings)