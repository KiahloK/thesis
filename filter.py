import json
import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if t]


def _endpoint_text(method: str, path: str, operation: dict) -> str:
    parts = [
        method,
        re.sub(r'[/_\-{}]', ' ', path),
    ]
    for key in ('operationId', 'summary', 'description'):
        if val := operation.get(key):
            parts.append(str(val))
    for param in operation.get('parameters', []):
        if name := param.get('name'):
            parts.append(name)
        if desc := param.get('description'):
            parts.append(desc)
    for tag in operation.get('tags', []):
        parts.append(tag)
    if desc := operation.get('requestBody', {}).get('description'):
        parts.append(desc)
    return ' '.join(parts)


def filter_services(services: list[str], query: str, top_k: int = 5) -> list[str]:
    """Return pruned OpenAPI JSON strings keeping only the top_k most query-relevant endpoints,
    ranked globally across all given services (not per service) - so a query touching five
    services still gets a total budget of top_k endpoints, not top_k per service.

    Endpoints are scored with BM25 against the query. The info/servers/components blocks are
    preserved so the model still has base URLs and shared schemas. Services that cannot be
    parsed, or have no paths, are returned unchanged.
    """
    if not services:
        return services

    query_tokens = _tokenize(query)

    specs: list[dict | None] = []
    all_endpoints: list[tuple[int, str, str, dict]] = []  # (service_index, method, path, operation)
    for si, service_json in enumerate(services):
        try:
            spec = json.loads(service_json)
        except (json.JSONDecodeError, ValueError):
            specs.append(None)
            continue
        specs.append(spec)
        for path, methods in spec.get('paths', {}).items():
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    all_endpoints.append((si, method.upper(), path, operation))

    if len(all_endpoints) <= top_k:
        # Nothing to prune globally
        return services

    corpus = [_tokenize(_endpoint_text(m, p, op)) for _, m, p, op in all_endpoints]
    scores = BM25Okapi(corpus).get_scores(query_tokens)

    top_indices = set(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )

    selected_paths: dict[int, dict] = {}
    for i, (si, method, path, operation) in enumerate(all_endpoints):
        if i in top_indices:
            selected_paths.setdefault(si, {}).setdefault(path, {})[method.lower()] = operation

    pruned: list[str] = []
    for si, service_json in enumerate(services):
        spec = specs[si]
        if spec is   or not spec.get('paths'):
            pruned.append(service_json)
            continue
        pruned_spec = {k: v for k, v in spec.items() if k != 'paths'}
        pruned_spec['paths'] = selected_paths.get(si, {})
        pruned.append(json.dumps(pruned_spec, ensure_ascii=False))

    return pruned
