from pathlib import Path
import json
from typing import Tuple, List, Optional


def load_benchmarks(benchmark_dir: str | Path, types: List[str], limit: Optional[int], query_limit: Optional[int]) -> Tuple[List[dict], int, int]:
    """Load benchmark sets from the given directory.

    Returns (benchmark_sets, total_available, total_queries)
    """
    benchmark_dir = Path(benchmark_dir)
    benchmark_sets = []

    for benchmark_type in types:
        type_dir = benchmark_dir / benchmark_type
        if not type_dir.exists():
            continue
        sectors = sorted(p for p in type_dir.iterdir() if p.is_dir())
        slice_sectors = sectors[:limit] if limit is not None else sectors
        for sector_path in slice_sectors:
            queries_path = sector_path / "queries.json"
            if not queries_path.exists():
                continue
            queries = json.loads(queries_path.read_text()).get("queries", [])
            services = []
            service_files = []
            for p in sector_path.rglob("openapi.json"):
                services.append(p.read_text())
                try:
                    service_files.append(str(p.relative_to(benchmark_dir)))
                except Exception:
                    service_files.append(str(p))
            benchmark_sets.append({
                "name": sector_path.name,
                "type": benchmark_type,
                "services": services,
                "service_files": service_files,
                "queries": queries[:query_limit] if query_limit is not None else queries,
            })

    total_available = sum(
        len([p for p in (benchmark_dir / t).iterdir() if p.is_dir()])
        for t in types
        if (benchmark_dir / t).exists()
    )
    total_queries = sum(len(b["queries"]) for b in benchmark_sets)
    return benchmark_sets, total_available, total_queries
