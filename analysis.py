from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Literal, Sequence
from urllib.parse import urlparse


AnalysisType = Literal["ast", "ruff", "httpretty", "ast_spec", "openapi"]
DEFAULT_ANALYSES: tuple[AnalysisType, ...] = ("ast", "ruff", "httpretty")
_ALL_ANALYSIS_TYPES: frozenset[str] = frozenset({"ast", "ruff", "httpretty", "ast_spec", "openapi"})


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
        if choice not in _ALL_ANALYSIS_TYPES:
            raise ValueError(
                f"Unknown analysis type '{analysis}'. Use one or more of: {', '.join(sorted(_ALL_ANALYSIS_TYPES))}."
            )
        if choice not in normalized:
            normalized.append(choice)
    return tuple(normalized)


# ---------------------------------------------------------------------------
# Helpers for OpenAPI spec loading and path matching
# ---------------------------------------------------------------------------

def _load_openapi_specs(service_files: list[str], benchmark_dir: str | Path) -> list[dict]:
    """Load and parse each OpenAPI JSON file relative to benchmark_dir."""
    specs = []
    bdir = Path(benchmark_dir)
    for sf in service_files:
        path = bdir / sf
        if path.exists():
            try:
                specs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    return specs


def _spec_endpoints(spec: dict) -> list[tuple[str, str, list[str]]]:
    """Return (METHOD, path_template, required_body_fields) for every operation in a spec."""
    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue
            schema = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            required = schema.get("required", [])
            endpoints.append((method.upper(), path, required))
    return endpoints


def _normalize_path(path: str) -> str:
    """Strip version prefix (e.g. /v1, /v2) from a path, matching evaluate.py behaviour."""
    return re.sub(r"^/v\d+", "", path)


def _strip_to_path(url: str) -> str:
    """Extract just the path component from a full or partial URL."""
    parsed = urlparse(url)
    return parsed.path if parsed.path else url


def _path_matches(concrete: str, template: str) -> bool:
    """Segment-by-segment comparison; {…} in either side is treated as a wildcard.

    A template wildcard segment must be filled by a non-empty concrete value - an
    empty segment (e.g. `/artists//top-tracks`, from a path param that resolved to
    "" or None at runtime) does not count as a match.
    """
    concrete = concrete.split("?")[0]
    c_segs = concrete.split("/")
    t_segs = template.split("/")
    if len(c_segs) != len(t_segs):
        return False
    for cs, ts in zip(c_segs, t_segs):
        is_wildcard = (cs.startswith("{") and cs.endswith("}")) or (ts.startswith("{") and ts.endswith("}"))
        if is_wildcard:
            if ts.startswith("{") and ts.endswith("}") and not cs:
                return False
            continue
        if cs != ts:
            return False
    return True


def _empty_param_template(concrete: str, endpoints: list[tuple[str, str, list[str]]], method: str) -> str | None:
    """If `concrete` would match one of `endpoints` (same method) except that an empty
    string was substituted for a path parameter, return that endpoint's template path.
    Used to give a specific, actionable finding instead of a generic "unknown endpoint".
    """
    concrete = concrete.split("?")[0]
    c_segs = concrete.split("/")
    for ep_method, ep_path, _required in endpoints:
        if ep_method != method:
            continue
        t_segs = ep_path.split("/")
        if len(c_segs) != len(t_segs):
            continue
        saw_empty_wildcard = False
        ok = True
        for cs, ts in zip(c_segs, t_segs):
            is_template_wildcard = ts.startswith("{") and ts.endswith("}")
            if is_template_wildcard:
                if not cs:
                    saw_empty_wildcard = True
                continue
            if cs != ts:
                ok = False
                break
        if ok and saw_empty_wildcard:
            return ep_path
    return None


# ---------------------------------------------------------------------------
# AST visitor for requests.METHOD(...) calls
# ---------------------------------------------------------------------------

class _RequestCallVisitor(ast.NodeVisitor):
    """Extract (METHOD, url_or_None, body_keys_or_None) from requests.* calls."""

    HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, list[str] | None]] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in self.HTTP_METHODS
            and isinstance(func.value, ast.Name)
            and func.value.id == "requests"
        ):
            method = func.attr.upper()
            url = self._resolve_str(node.args[0]) if node.args else None
            body_keys = self._resolve_body_keys(node.keywords)
            self.calls.append((method, url, body_keys))
        self.generic_visit(node)

    def _resolve_str(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                else:
                    parts.append("{...}")
            return "".join(parts)
        return None

    def _resolve_body_keys(self, keywords: list[ast.keyword]) -> list[str] | None:
        for kw in keywords:
            if kw.arg in ("json", "data") and isinstance(kw.value, ast.Dict):
                keys: list[str] = []
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.append(k.value)
                return keys
        return None


# ---------------------------------------------------------------------------
# Existing analyses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# New analyses: AST+spec and OpenAPI runtime validation
# ---------------------------------------------------------------------------

def analyze_ast_spec(
    code: str,
    service_files: list[str] | None,
    benchmark_dir: str | Path = "./benchmark",
) -> list[str]:
    """Walk the AST for requests.METHOD(url) calls and validate each against the OpenAPI specs."""
    if not service_files:
        return ["ASTSpec: No service files provided — skipping spec validation."]

    specs = _load_openapi_specs(service_files, benchmark_dir)
    if not specs:
        return ["ASTSpec: Could not load any OpenAPI specs — skipping."]

    all_endpoints: list[tuple[str, str, list[str]]] = []
    for spec in specs:
        all_endpoints.extend(_spec_endpoints(spec))

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["ASTSpec: Cannot walk AST — syntax error (see AST findings)."]

    visitor = _RequestCallVisitor()
    visitor.visit(tree)

    if not visitor.calls:
        return ["ASTSpec: No requests calls found in code."]

    findings: list[str] = []
    for method, url, body_keys in visitor.calls:
        if url is None:
            findings.append(
                f"ASTSpec: Could not statically resolve URL for `requests.{method.lower()}(...)` call."
            )
            continue

        raw_path = _normalize_path(_strip_to_path(url))

        matched_methods: list[str] = []
        has_exact_method = False
        for ep_method, ep_path, ep_required in all_endpoints:
            if _path_matches(raw_path, ep_path):
                matched_methods.append(ep_method)
                if ep_method == method:
                    has_exact_method = True
                    if body_keys is not None and ep_required:
                        missing = [r for r in ep_required if r not in body_keys]
                        if missing:
                            findings.append(
                                f"ASTSpec: `{method} {raw_path}` is missing required body field(s): {', '.join(missing)}."
                            )

        if not matched_methods:
            findings.append(
                f"ASTSpec: `{method} {raw_path}` does not match any endpoint in the provided specs."
            )
        elif not has_exact_method:
            findings.append(
                f"ASTSpec: `{method} {raw_path}` — path exists but only as {'/'.join(matched_methods)}."
            )

    if not findings:
        findings.append("ASTSpec: All statically-resolved requests calls match a spec endpoint.")
    return findings


# Extended httpretty runner that also captures request body per call.
_OPENAPI_RUNNER = textwrap.dedent("""\
    import re, sys, json as _json
    import httpretty
    from httpretty.core import fakesock

    fakesock.socket.shutdown = lambda self, *a, **kw: None

    _calls = []

    def _handler(request, uri, response_headers):
        try:
            body_str = request.body.decode("utf-8", errors="replace") if request.body else ""
        except Exception:
            body_str = ""
        _calls.append((request.method, uri, body_str))
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
        for _method, _uri, _body in _calls:
            print(f"HTTP_CALL:{_method}\\t{_uri}\\t{_body}")
        httpretty.disable()
        httpretty.reset()
""")


def analyze_openapi(
    code: str,
    service_files: list[str] | None,
    benchmark_dir: str | Path = "./benchmark",
) -> list[str]:
    """Run code with httpretty, then validate each captured HTTP call against the OpenAPI specs."""
    if not service_files:
        return ["OpenAPI: No service files provided — skipping validation."]

    specs = _load_openapi_specs(service_files, benchmark_dir)
    if not specs:
        return ["OpenAPI: Could not load any OpenAPI specs — skipping."]

    all_endpoints: list[tuple[str, str, list[str]]] = []
    for spec in specs:
        all_endpoints.extend(_spec_endpoints(spec))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "generated.py"
        runner_path = tmp / "_openapi_runner.py"
        code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_OPENAPI_RUNNER, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(code_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return ["OpenAPI: Execution timed out after 10 seconds."]

    findings: list[str] = []
    for line in completed.stderr.splitlines():
        if line.startswith("RUNTIME_ERROR:"):
            findings.append(f"OpenAPI: Runtime error — {line[len('RUNTIME_ERROR:'):]}")

    http_calls: list[tuple[str, str, str]] = []
    for line in completed.stdout.splitlines():
        if line.startswith("HTTP_CALL:"):
            parts = line[len("HTTP_CALL:"):].split("\t", 2)
            if len(parts) >= 2:
                method, uri = parts[0], parts[1]
                body_str = parts[2] if len(parts) > 2 else ""
                http_calls.append((method, uri, body_str))

    if not http_calls:
        if not findings:
            findings.append("OpenAPI: Code executed without HTTP calls.")
        return findings

    for method, uri, body_str in http_calls:
        path = _normalize_path(_strip_to_path(uri))

        body_keys: list[str] | None = None
        if body_str.strip():
            try:
                body_data = json.loads(body_str)
                if isinstance(body_data, dict):
                    body_keys = list(body_data.keys())
            except Exception:
                pass

        matched_methods: list[str] = []
        has_exact_method = False
        for ep_method, ep_path, ep_required in all_endpoints:
            if _path_matches(path, ep_path):
                matched_methods.append(ep_method)
                if ep_method == method:
                    has_exact_method = True
                    if body_keys is not None and ep_required:
                        missing = [r for r in ep_required if r not in body_keys]
                        if missing:
                            findings.append(
                                f"OpenAPI: `{method} {path}` is missing required body field(s): {', '.join(missing)}."
                            )

        if not matched_methods:
            empty_template = _empty_param_template(path, all_endpoints, method)
            if empty_template:
                findings.append(
                    f"OpenAPI: `{method} {path}` looks like a call to `{method} {empty_template}` with an "
                    "empty path parameter (note the `//` or trailing `/`). The ID/value being substituted "
                    "into the URL is empty or None - check where it's extracted from the prior response."
                )
            else:
                findings.append(f"OpenAPI: `{method} {path}` is not defined in any provided spec.")
        elif not has_exact_method:
            findings.append(
                f"OpenAPI: `{method} {path}` — path exists but only as {'/'.join(matched_methods)}."
            )

    if not findings:
        findings.append("OpenAPI: All runtime HTTP calls match a spec endpoint.")
    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(
    code: str,
    analyses: Sequence[str] | str | None = None,
    *,
    service_files: list[str] | None = None,
    benchmark_dir: str | Path = "./benchmark",
) -> str:
    """Return a short report for the selected analyses.

    When *analyses* is None and *service_files* is provided, the two spec-aware
    analyses (ast_spec, openapi) are added on top of the defaults automatically.
    """
    selected = _normalize_analyses(analyses)
    if analyses is None and service_files:
        selected = selected + ("ast_spec", "openapi")

    findings: list[str] = []

    if "ast" in selected:
        findings.extend(analyze_ast(code))
    if "ruff" in selected:
        findings.extend(analyze_ruff(code))
    if "httpretty" in selected:
        findings.extend(analyze_httpretty(code))
    if "ast_spec" in selected:
        findings.extend(analyze_ast_spec(code, service_files, benchmark_dir))
    if "openapi" in selected:
        findings.extend(analyze_openapi(code, service_files, benchmark_dir))

    return "\n".join(findings)
