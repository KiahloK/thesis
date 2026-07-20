import json
import threading

import faiss
from sentence_transformers import SentenceTransformer

from filter import _endpoint_text

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBED_DIM = 384

_model: SentenceTransformer | None = None

# PyTorch/FAISS on this platform are not safe to call concurrently from multiple threads
# (racing native OpenMP init/compute can segfault the process) - serialize all embedding
# and index work behind one lock. LLM network calls (the actual expensive, parallelizable
# part of the pipeline) are unaffected since they never touch this lock.
_compute_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def filter_services_rag(services: list[str], query: str, top_k: int = 5) -> list[str]:
    """Return pruned OpenAPI JSON strings keeping only the top_k most query-relevant endpoints per service.

    Endpoints are scored by cosine similarity between their embedding and the query
    embedding (both encoded with a sentence-transformers model), using a per-service
    FAISS flat index. The info/servers/components blocks are preserved so the model
    still has base URLs and shared schemas. Services that cannot be parsed are
    returned unchanged.
    """
    if not services:
        return services

    pruned: list[str] = []

    with _compute_lock:
        model = _get_model()
        query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

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

            texts = [_endpoint_text(m, p, op) for m, p, op in endpoints]
            embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

            index = faiss.IndexFlatIP(_EMBED_DIM)
            index.add(embeddings)
            _, top_indices_arr = index.search(query_vector, top_k)
            top_indices = set(int(i) for i in top_indices_arr[0] if i != -1)

            pruned_paths: dict = {}
            for i, (method, path, operation) in enumerate(endpoints):
                if i in top_indices:
                    pruned_paths.setdefault(path, {})[method.lower()] = operation

            pruned_spec = {k: v for k, v in spec.items() if k != 'paths'}
            pruned_spec['paths'] = pruned_paths
            pruned.append(json.dumps(pruned_spec, ensure_ascii=False))

    return pruned
