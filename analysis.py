from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Literal, Sequence


AnalysisType = Literal["ast", "ruff", "httpretty"]
DEFAULT_ANALYSES: tuple[AnalysisType, ...] = ("ast", "ruff", "httpretty")


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
                f"Unknown analysis type '{analysis}'. Use one or more of: ast, ruff, httpretty."
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


_HTTPRETTY_RUNNER = textwrap.dedent("""\
    import re, sys
    import httpretty
    from httpretty.core import fakesock

    # fakesock has no shutdown method; its __getattr__ raises UnmockedError for
    # any unknown attribute, causing requests' connection cleanup to throw.
    # Patching it to a no-op lets multi-request code run uninterrupted.
    fakesock.socket.shutdown = lambda self, *a, **kw: None

    _calls = []

    def _handler(request, uri, response_headers):
        _calls.append(f"{request.method} {uri}")
        return [200, response_headers, b"{}"]

    httpretty.enable(allow_net_connect=False, verbose=False)
    for _method in (
        httpretty.GET, httpretty.POST, httpretty.PUT,
        httpretty.PATCH, httpretty.DELETE, httpretty.HEAD, httpretty.OPTIONS,
    ):
        httpretty.register_uri(_method, re.compile(r".*"), body=_handler)

    try:
        with open(sys.argv[1], encoding="utf-8") as _f:
            exec(compile(_f.read(), sys.argv[1], "exec"), {"__name__": "__main__"})
    except Exception as exc:
        print(f"RUNTIME_ERROR:{exc}", file=sys.stderr)
    finally:
        for _call in _calls:
            print(f"HTTP_CALL:{_call}")
        httpretty.disable()
        httpretty.reset()
""")


def analyze_httpretty(code: str, *, filename_hint: str = "generated.py") -> list[str]:
    """Execute the code with httpretty intercepting all HTTP calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / filename_hint
        runner_path = tmp / "_runner.py"
        code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_HTTPRETTY_RUNNER, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(code_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return ["Httpretty: Execution timed out after 10 seconds."]

    findings: list[str] = []
    for line in completed.stdout.splitlines():
        if line.startswith("HTTP_CALL:"):
            findings.append(f"Httpretty: {line[len('HTTP_CALL:'):]}")
    for line in completed.stderr.splitlines():
        if line.startswith("RUNTIME_ERROR:"):
            findings.append(f"Httpretty: Runtime error — {line[len('RUNTIME_ERROR:'):]}")

    if not findings:
        if completed.returncode == 0:
            return ["Httpretty: Code executed without HTTP calls or errors."]
        return ["Httpretty: Execution failed without output."]

    return findings


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

    if "httpretty" in selected:
        findings.extend(analyze_httpretty(code))

    return "\n".join(findings)