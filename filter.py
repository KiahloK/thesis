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
    """Return pruned OpenAPI JSON strings keeping only the top_k most query-relevant endpoints per service.

    Endpoints are scored with BM25 against the query. The info/servers/components
    blocks are preserved so the model still has base URLs and shared schemas.
    Services that cannot be parsed are returned unchanged.
    """
    if not services:
        return services

    query_tokens = _tokenize(query)
    pruned: list[str] = []

    for service_json in services:
        try:
            spec = json.loads(service_json)
        except (json.JSONDecodeError, ValueError):
            pruned.append(service_json)
            continue

        paths = spec.get('paths', {})
        if not paths:
            pruned.append(service_json)
            continue

        # Flatten to (method, path, operation) triples
        endpoints: list[tuple[str, str, dict]] = []
        for path, methods in paths.items():
            for method, operation in methods.items():
                if isinstance(operation, dict):
                    endpoints.append((method.upper(), path, operation))

        if len(endpoints) <= top_k:
            # Nothing to prune
            pruned.append(service_json)
            continue

        corpus = [_tokenize(_endpoint_text(m, p, op)) for m, p, op in endpoints]
        scores = BM25Okapi(corpus).get_scores(query_tokens)

        top_indices = set(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )

        pruned_paths: dict = {}
        for i, (method, path, operation) in enumerate(endpoints):
            if i in top_indices:
                pruned_paths.setdefault(path, {})[method.lower()] = operation

        pruned_spec = {k: v for k, v in spec.items() if k != 'paths'}
        pruned_spec['paths'] = pruned_paths
        pruned.append(json.dumps(pruned_spec, ensure_ascii=False))

    return pruned
