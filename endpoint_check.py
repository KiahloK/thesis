import json
import re

from socbenchsc.analysis import Analysis

from evaluate import _matches_template, _normalize_endpoint, server_base_paths
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
Query: Book a table for two at the restaurant with the best rating.
Endpoint: POST /reservations - Create a table reservation at a restaurant
Yes
Endpoint: DELETE /reservations/{{reservationId}} - Cancel an existing reservation
No

INSTRUCTIONS:
You are an expert judge determining if an endpoint is needed to fulfill a query. Check if an endpoint from the list of endpoints is necessary to fulfill the query: answer "Yes" only if the endpoint directly performs a step the query asks for. Do not format the output as you are in an automated setting.'''


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


def _called_endpoints(generated_code: str, services: list[str]) -> tuple[set[str], set[str]]:
    """Statically extract the endpoints `generated_code` calls and fold each one onto its matching
    "METHOD /path" template in `services`. Returns (called, matched): `called` holds every call
    (unmatched ones as extracted), `matched` only the calls that map onto an endpoint in `services`.
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

    specs = _parse_services(services)
    base_paths = server_base_paths(specs)
    candidates = {c: _normalize_endpoint(c) for c in _all_endpoints(specs)}

    # Fold each extracted (possibly concrete, e.g. path params filled in) endpoint onto its
    # matching candidate template, so e.g. "GET /tracks/abc123" and "GET /tracks/{id}"
    # count as the same call instead of showing up as both missing and additional.
    called = set()
    matched = set()
    for ext in extracted:
        norm_ext = _normalize_endpoint(ext, base_paths)
        match = next((c for c, norm_c in candidates.items() if _matches_template(norm_ext, norm_c)), None)
        called.add(match if match else ext)
        if match:
            matched.add(match)
    return called, matched


def called_endpoints(generated_code: str, services: list[str]) -> list[str]:
    """Endpoints of `services` (as "METHOD /path" templates) that `generated_code` calls - used to
    keep the retrieval filter from pruning an endpoint the code under refinement already uses."""
    return sorted(_called_endpoints(generated_code, services)[1])


def find_endpoint_issues(generated_code: str, services: list[str], query: str, model: str) -> dict:
    """Reference-free coverage check: determine which endpoints are necessary for `query` (via LLM
    judgment over `services`, no ground truth involved) and diff that against what the code
    actually calls (via the existing static AST analysis).

    Returns {'necessary': [...], 'called': [...], 'missing': [...], 'additional': [...]}.
    `missing` (necessary but not called) and `additional` (called but not necessary) are meant to
    be fed into a refinement prompt as actionable findings, the same way Ruff/spec-conformance
    findings already are. `additional` only covers endpoints present in `services`: a call to an
    endpoint the retrieval filter removed was never judged, so it can't be called unnecessary
    (filtering with always_keep=called_endpoints(...) keeps those in, so they get judged too).
    """
    called, judged_calls = _called_endpoints(generated_code, services)
    necessary = find_necessary_endpoints(services, query, model)
    necessary_set = set(necessary)

    return {
        'necessary': sorted(necessary_set),
        'called': sorted(called),
        'missing': sorted(necessary_set - called),
        'additional': sorted(judged_calls - necessary_set),
    }


# ---------------------------------------------------------------------------
# Step-wise check
# ---------------------------------------------------------------------------
# The per-endpoint check above answers Yes/No for every endpoint independently, so it cannot
# express "only one of these two alternatives is needed": on SOCBench-D it judged ~10 endpoints
# per query necessary where ~5 are expected, and its "possibly missing" list was 75% noise.
# The step-wise check instead splits the query into its steps and lets the LLM pick exactly one
# endpoint (or none) per step, which yields a step -> endpoint mapping the refinement can act on.

_STEPS_TEMPLATE = '''Split the following task into the individual steps it asks for. Each step is one action that a single API call could perform, e.g. retrieving, creating, updating or deleting something. Keep the order of the task. Do not add steps the task does not ask for, and do not merge two actions into one step.

Return one step per line, numbered "1.", "2.", ..., and nothing else.

Task:
{query}'''

_SELECT_TEMPLATE = '''You map one step of a task to the API endpoint that performs it.

Task:
{query}

Step:
{step}

Candidate endpoints:
{endpoints}

Answer with the one candidate endpoint that performs this step best, written exactly as "METHOD /path" from the list. If none of the candidates performs this step, answer NONE. Do not write anything else.'''

_STEP_LINE = re.compile(r'^\s*(?:step\s*)?\d+\s*[.):-]\s*(.+?)\s*$', re.IGNORECASE)


def _parse_steps(response: str) -> list[str]:
    """Numbered lines of `response`; if the model did not number them, every non-empty line."""
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    numbered = [m.group(1) for line in lines if (m := _STEP_LINE.match(line))]
    return numbered or [line.lstrip('-* ').strip() for line in lines]


def _parse_selection(response: str, labels: list[str]) -> str | None:
    """The candidate label named in `response`, or None. If several labels occur in the first
    line (e.g. 'GET /alerts' inside 'GET /alerts/{id}'), the longest one wins."""
    first_line = next((line.strip().strip('`*"\'') for line in response.splitlines() if line.strip()), '')
    if not first_line or first_line.upper().startswith('NONE'):
        return None
    contained = [label for label in labels if label in first_line]
    return max(contained, key=len) if contained else None


def _endpoint_lines(specs: list[dict]) -> list[tuple[str, str]]:
    """(label, "label - summary. description") per operation; the description helps the LLM
    tell apart near-duplicate endpoints from different services."""
    endpoints = []
    for spec in specs:
        for path, methods in spec.get('paths', {}).items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    label = f"{method.upper()} {path}"
                    parts = [str(operation.get(key) or '').strip() for key in ('summary', 'description')]
                    text = ' '.join(' '.join(p for p in dict.fromkeys(parts) if p).split())
                    endpoints.append((label, f"{label} - {text}" if text else label))
    return endpoints


def find_step_issues(generated_code: str, services: list[str], query: str, model: str) -> dict:
    """Reference-free, step-wise coverage check: split `query` into steps, map each step to one
    endpoint of `services` via the LLM, and diff that mapping against the calls of the code.

    Returns {'steps': [{'step', 'endpoint', 'called'}], 'necessary', 'called', 'missing',
    'additional'} - the last four with the same meaning as in find_endpoint_issues(), so runs of
    both modes can be analysed with the same tools.
    """
    called, judged_calls = _called_endpoints(generated_code, services)
    candidates = _endpoint_lines(_parse_services(services))
    labels = [label for label, _ in candidates]

    response, _usage = call_llm_chat([{'role': 'user', 'content': _STEPS_TEMPLATE.format(query=query)}], model)
    steps = _parse_steps(response)

    endpoints_block = '\n'.join(line for _, line in candidates)
    mapping = []
    for step in steps:
        prompt = _SELECT_TEMPLATE.format(query=query, step=step, endpoints=endpoints_block)
        response, _usage = call_llm_chat([{'role': 'user', 'content': prompt}], model)
        endpoint = _parse_selection(response, labels)
        mapping.append({'step': step, 'endpoint': endpoint, 'called': endpoint in called if endpoint else None})

    necessary = {m['endpoint'] for m in mapping if m['endpoint']}
    return {
        'steps': mapping,
        'necessary': sorted(necessary),
        'called': sorted(called),
        'missing': sorted(necessary - called),
        'additional': sorted(judged_calls - necessary),
    }


def format_step_issues_for_prompt(issues: dict) -> str:
    """Render find_step_issues() output as findings text for the refinement prompt."""
    if not issues['steps']:
        return ''
    lines = ["Step coverage (an automated check split the Task into steps and mapped each step to the endpoint that performs it):"]
    for i, m in enumerate(issues['steps'], start=1):
        if m['endpoint'] is None:
            lines.append(f"{i}. {m['step']} -> no endpoint in Source performs this step")
        elif m['called']:
            lines.append(f"{i}. {m['step']} -> {m['endpoint']} (called)")
        else:
            lines.append(f"{i}. {m['step']} -> {m['endpoint']} (NOT called: use this endpoint for this step)")
    if issues['additional']:
        unmapped = [str(i) for i, m in enumerate(issues['steps'], start=1) if m['endpoint'] is None]
        if unmapped:
            # A step the check could not map may well be served by one of these calls, so they
            # must not be presented as definitely unnecessary.
            lines.append(
                f"Calls not mapped to any step: {', '.join(issues['additional'])}. Keep such a call only "
                f"if it performs step {', '.join(unmapped)}; otherwise remove it or replace it with the "
                "endpoint mapped above."
            )
        else:
            lines.append(
                "Calls that perform no step of the Task (remove them, or replace them with the endpoint "
                "mapped above): " + ', '.join(issues['additional'])
            )
    return '\n'.join(lines)


def format_issues_for_prompt(issues: dict) -> str:
    """Render find_endpoint_issues() output as findings text to append to a refinement prompt."""
    lines = []
    if issues['missing']:
        lines.append(
            "Possibly missing endpoints (an automated check suggests the query may need these, but "
            "it is often wrong - add one only if it performs a step the Task explicitly asks for "
            "that the code does not cover yet): "
            + ", ".join(issues['missing'])
        )
    if issues['additional']:
        lines.append(
            "Unnecessary endpoint calls (the query does not require these, remove them): "
            + ", ".join(issues['additional'])
        )
    return "\n".join(lines)
