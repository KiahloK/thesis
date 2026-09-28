#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --job-name=refined
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Runs refined.ipynb headless: initial call -> evaluate -> analyze -> RAG filter +
# endpoint check (Idea 2) -> refinement -> evaluate. Results land in output/ as usual;
# the executed notebook (with all printed output) is saved as slurm-refined-<jobid>.ipynb.
#
# Every notebook setting can be overridden at submission time, e.g.
#   QUERY_LIMIT=50 FILTER_MODE=bm25 sbatch slurm/refined.sbatch

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

export LOCAL_MODEL_ID="${LOCAL_MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
export LOCAL_MAX_NEW_TOKENS="${LOCAL_MAX_NEW_TOKENS:-1024}"
export LOCAL_DO_SAMPLE="${LOCAL_DO_SAMPLE:-0}"
export LOCAL_MAX_WORKERS="${LOCAL_MAX_WORKERS:-1}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

# Notebook configuration (see the Configuration cell in refined.ipynb).
export BENCHMARK_TYPES="${BENCHMARK_TYPES:-socbenchd_1}"
export QUERY_LIMIT="${QUERY_LIMIT:-10}"
export LOAD_BASELINE="${LOAD_BASELINE:-0}"
export FILTER_SERVICES="${FILTER_SERVICES:-1}"
export FILTER_MODE="${FILTER_MODE:-rag}"
export TOP_K="${TOP_K:-10}"
export ENDPOINT_CHECK="${ENDPOINT_CHECK:-1}"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
# analysis.py looks up `ruff` on PATH; the venv's bin dir is not on PATH without activation.
export PATH="$(dirname "$PYTHON"):$PATH"

if [[ ! -d benchmark/"${BENCHMARK_TYPES%%,*}" ]]; then
    echo "Benchmark data not found under $PROJECT_ROOT/benchmark (it is gitignored - copy it over)." >&2
    exit 1
fi

echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Model: $LOCAL_MODEL_ID | HF_HOME: $HF_HOME | HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}"
echo "Benchmarks: $BENCHMARK_TYPES | Queries: $QUERY_LIMIT | Load baseline: $LOAD_BASELINE"
echo "Filter: $FILTER_SERVICES ($FILTER_MODE, top_k=$TOP_K) | Endpoint check: $ENDPOINT_CHECK"
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"
command -v ruff

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_GPU:-24}" \
    "$PYTHON" -m nbconvert --to notebook --execute refined.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$PROJECT_ROOT" \
    --output "slurm-${SLURM_JOB_NAME:-refined}-${SLURM_JOB_ID:-local}.ipynb"
