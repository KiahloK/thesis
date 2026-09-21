import json
import threading

import faiss
from sentence_transformers import SentenceTransformer

from filter import _endpoint_text

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384

_model: SentenceTransformer | None = None

_compute_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def filter_services_rag(services: list[str], query: str, top_k: int = 5) -> list[str]:
    """Return pruned OpenAPI JSON strings keeping only the top_k most query-relevant endpoints,
    ranked globally across all given services (not per service) - so a query touching five
    services still gets a total budget of top_k endpoints, not top_k per service.

    Endpoints are scored by cosine similarity between their embedding and the query embedding
    (both encoded with a sentence-transformers model), using a single FAISS flat index over all
    endpoints from all services. The info/servers/components blocks are preserved so the model
    still has base URLs and shared schemas. Services that cannot be parsed, or have no paths,
    are returned unchanged.
    """
    if not services:
        return services

    with _compute_lock:
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
        # TODO: pydantic model
        model = _get_model()
        query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
        texts = [_endpoint_text(m, p, op) for _, m, p, op in all_endpoints]
        embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

        index = faiss.IndexFlatIP(_EMBED_DIM)
        index.add(embeddings)
        _, top_indices_arr = index.search(query_vector, top_k)
        top_indices = set(int(i) for i in top_indices_arr[0] if i != -1)

        selected_paths: dict[int, dict] = {}
        for i, (si, method, path, operation) in enumerate(all_endpoints):
            if i in top_indices:
                selected_paths.setdefault(si, {}).setdefault(path, {})[method.lower()] = operation

        pruned: list[str] = []
        for si, service_json in enumerate(services):
            spec = specs[si]
            if spec is None or not spec.get('paths'):
                pruned.append(service_json)
                continue
            pruned_spec = {k: v for k, v in spec.items() if k != 'paths'}
            pruned_spec['paths'] = selected_paths.get(si, {})
            pruned.append(json.dumps(pruned_spec, ensure_ascii=False))

    return pruned
