import json

from socbenchsc.analysis import Analysis

from evaluate import _matches_template, _normalize_endpoint
from llm import call_llm_chat

_TEMPLATE = '''SUMMARY:
Check endpoints if they are necessary.

DOCUMENT:
Services:
{services}

Query:
{query}

List of endpoints:
{endpoints}

TASK:
You are given a query and a set of OpenAPI specifications. Check for each of the given endpoints in the list of endpoints if it is required to fulfill the task. Return "Yes" if it is required or "No" if not.

EXAMPLE
Yes
No

INSTRUCTIONS:
You are an expert judge determining if an endpoint is needed to fulfill a query. Check if an endpoint from the list of endpoints is necessary to fulfill the query, or if it is necessary to retrieve parameters for another endpoint in the list. Do not format the output as you are in an automated setting.'''


def _parse_services(services: list[str]) -> list[dict]:
    specs = []
    for service_json in services:
        try:
            specs.append(json.loads(service_json))
        except (json.JSONDecodeError, ValueError):
            continue
    return specs


def _all_endpoints(specs: list[dict]) -> list[str]:
    labels = []
    for spec in specs:
        for path, methods in spec.get('paths', {}).items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    labels.append(f"{method.upper()} {path}")
    return labels


def _operation_description(operation: dict) -> str:
    text = operation.get('summary') or operation.get('description') or ''
    return ' '.join(str(text).split())


def _all_endpoints_described(specs: list[dict]) -> list[tuple[str, str]]:
    """Like _all_endpoints, but pairs each label with a display string that appends the
    operation's summary/description, so the LLM sees what an endpoint does, not just its path.
    The bare label stays the key for matching against extracted calls."""
    endpoints = []
    for spec in specs:
        for path, methods in spec.get('paths', {}).items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    label = f"{method.upper()} {path}"
                    desc = _operation_description(operation)
                    endpoints.append((label, f"{label} - {desc}" if desc else label))
    return endpoints


def find_necessary_endpoints(services: list[str], query: str, model: str) -> list[str]:
    """Reference-free: ask the LLM which endpoints in `services` are necessary to fulfill `query`.

    """
    specs = _parse_services(services)
    candidates = _all_endpoints_described(specs)
    if not candidates:
        return []

    services_block = "\n---\n".join(services)
    endpoints_block = "\n".join(display for _, display in candidates)
    instance = _TEMPLATE.format(services=services_block, query=query, endpoints=endpoints_block)
    messages = [
        {"role": "user", "content": instance},
        {"role": "assistant", "content": "Ok, please provide me the first endpoint."},
    ]

    necessary = []
    for label, display in candidates:
        messages.append({"role": "user", "content": display})
        response, _usage = call_llm_chat(messages, model)
        messages.append({"role": "assistant", "content": response})
        if response.strip().startswith("Yes") or response.strip().startswith("**Yes**"):
            necessary.append(label)

    return necessary


def find_endpoint_issues(generated_code: str, services: list[str], query: str, model: str) -> dict:
    """Reference-free coverage check: determine which endpoints are necessary for `query` (via LLM
    judgment over `services`, no ground truth involved) and diff that against what the code
    actually calls (via the existing static AST analysis).

    Returns {'necessary': [...], 'called': [...], 'missing': [...], 'additional': [...]}.
    `missing` (necessary but not called) and `additional` (called but not necessary) are meant to
    be fed into a refinement prompt as actionable findings, the same way Ruff/spec-conformance
    findings already are. `additional` only covers endpoints present in `services`: a call to an
    endpoint the retrieval filter removed was never judged, so it can't be called unnecessary.
    """
    # Analysis only detects requests calls reachable from a top-level compose() invocation,
    # so code with the call stripped (e.g. the refinement prompt's "original code") would
    # otherwise always extract as empty, making every necessary endpoint look "missing".
    code_for_analysis = generated_code
    if not code_for_analysis.rstrip().endswith('compose()'):
        code_for_analysis += '\n\ncompose()'

    try:
        extracted = Analysis(code_for_analysis).perform_analysis()
    except SyntaxError:
        extracted = set()

    necessary = find_necessary_endpoints(services, query, model)
    necessary_set = set(necessary)
    candidates = {c: _normalize_endpoint(c) for c in _all_endpoints(_parse_services(services))}

    # Fold each extracted (possibly concrete, e.g. path params filled in) endpoint onto its
    # matching candidate template, so e.g. "GET /tracks/abc123" and "GET /tracks/{id}"
    # count as the same call instead of showing up as both missing and additional.
    called = set()
    judged_calls = set()
    for ext in extracted:
        norm_ext = _normalize_endpoint(ext)
        match = next((c for c, norm_c in candidates.items() if _matches_template(norm_ext, norm_c)), None)
        called.add(match if match else ext)
        if match:
            judged_calls.add(match)

    return {
        'necessary': sorted(necessary_set),
        'called': sorted(called),
        'missing': sorted(necessary_set - called),
        'additional': sorted(judged_calls - necessary_set),
    }


def format_issues_for_prompt(issues: dict) -> str:
    """Render find_endpoint_issues() output as findings text to append to a refinement prompt."""
    lines = []
    if issues['missing']:
        lines.append(
            "Missing required endpoints (the query needs these but the code never calls them): "
            + ", ".join(issues['missing'])
        )
    if issues['additional']:
        lines.append(
            "Unnecessary endpoint calls (the query does not require these, remove them): "
            + ", ".join(issues['additional'])
        )
    return "\n".join(lines)
