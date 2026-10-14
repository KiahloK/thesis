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
# To split a run across dev-queue jobs, run chunks with QUERY_OFFSET=0, 10, 20, ... and QUERY_LIMIT=10.
# This also works with LOAD_BASELINE=1 BASELINE_DIR=output/<run>: each chunk then refines a slice
# of that run's stored initial code.

set -euo pipefail

# Print the job's wall-clock time when the script ends - also on errors and on the time limit
# (Slurm sends SIGTERM before killing the job). Compare with `sacct -j <jobid> --format=JobID,Elapsed,State`.
JOB_STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
trap 'echo "Job started $JOB_STARTED, ended $(date "+%Y-%m-%d %H:%M:%S"), wall time $((SECONDS / 60))m $((SECONDS % 60))s"' EXIT
trap 'exit 143' TERM

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

# The venv is built on this module's Python (socbenchsc needs Python >= 3.10).
PYTHON_MODULE="${PYTHON_MODULE:-devel/python/3.12.3-gnu-14.2}"
if command -v module >/dev/null 2>&1; then
    set +u  # Lmod scripts may read unset variables
    module load "$PYTHON_MODULE"
    set -u
fi

export LOCAL_MODEL_ID="${LOCAL_MODEL_ID:-Qwen/Qwen3.8-27B}"
export LOCAL_MAX_NEW_TOKENS="${LOCAL_MAX_NEW_TOKENS:-1024}"
export LOCAL_DO_SAMPLE="${LOCAL_DO_SAMPLE:-0}"
# Reasoning per call type: off|low|medium|high (thinking budget 1024/4096/16384 tokens, see llm.py).
# LOCAL_REASONING sets all three; LOCAL_REASONING_GENERATE/_REFINE/_CHECK override it per call type.
# Only takes effect for models with a thinking mode (e.g. Qwen3-style templates). Reasoning makes
# calls much slower - use smaller chunks (e.g. QUERY_LIMIT=3) to stay within the 30-minute limit.
export LOCAL_REASONING="${LOCAL_REASONING:-off}"
export LOCAL_REASONING_GENERATE="${LOCAL_REASONING_GENERATE:-$LOCAL_REASONING}"
export LOCAL_REASONING_REFINE="${LOCAL_REASONING_REFINE:-$LOCAL_REASONING}"
export LOCAL_REASONING_CHECK="${LOCAL_REASONING_CHECK:-$LOCAL_REASONING}"
# Step-wise check: level of the single call that splits the query into steps (default: as CHECK).
export LOCAL_REASONING_CHECK_SPLIT="${LOCAL_REASONING_CHECK_SPLIT:-$LOCAL_REASONING_CHECK}"
export LOCAL_MAX_WORKERS="${LOCAL_MAX_WORKERS:-1}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

# Notebook configuration (see the Configuration cell in refined.ipynb).
export BENCHMARK_TYPES="${BENCHMARK_TYPES:-socbenchd_1}"
export QUERY_LIMIT="${QUERY_LIMIT:-10}"
export QUERY_OFFSET="${QUERY_OFFSET:-0}"
export LOAD_BASELINE="${LOAD_BASELINE:-0}"
export FILTER_SERVICES="${FILTER_SERVICES:-1}"
# BM25 at top_k=25 keeps half the spec and drops ~1% of expected endpoints on SOCBench-D1
# (RAG at the same top_k drops ~13%); see sweep_top_k_coverage.py.
export FILTER_MODE="${FILTER_MODE:-bm25}"
export TOP_K="${TOP_K:-25}"
# Also keep every endpoint the initial code already calls, on top of TOP_K, so the filter can't
# prune a call the model already made (and Idea 2 gets to judge it).
export KEEP_CALLED="${KEEP_CALLED:-1}"
export ENDPOINT_CHECK="${ENDPOINT_CHECK:-1}"
# "stepwise": split the query into steps, one endpoint per step; "per_endpoint": Yes/No per endpoint
# (the check used until 2026-10-13).
export ENDPOINT_CHECK_MODE="${ENDPOINT_CHECK_MODE:-stepwise}"
# Runtime analyses answer requests with bodies derived from the specs ("spec") or with {} ("empty",
# the behaviour before 2026-10-13).
export ANALYSIS_MOCK_RESPONSES="${ANALYSIS_MOCK_RESPONSES:-spec}"
# "repair": repair the code using the findings; "plan": rewrite compose() to follow the step-wise
# check's step -> endpoint plan (needs ENDPOINT_CHECK=1 and ENDPOINT_CHECK_MODE=stepwise).
export REFINE_MODE="${REFINE_MODE:-repair}"
# Analysis tools in the refinement prompt ("tools on/off"): all | none | comma-separated subset of
# ast,ruff,httpretty,ast_spec,openapi. Filtering and the endpoint check are not affected.
export ANALYSES="${ANALYSES:-all}"

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
echo "Reasoning: generate=$LOCAL_REASONING_GENERATE refine=$LOCAL_REASONING_REFINE check=$LOCAL_REASONING_CHECK check_split=$LOCAL_REASONING_CHECK_SPLIT"
echo "Benchmarks: $BENCHMARK_TYPES | Queries: $QUERY_LIMIT (offset $QUERY_OFFSET) | Load baseline: $LOAD_BASELINE ${BASELINE_DIR:-}"
echo "Filter: $FILTER_SERVICES ($FILTER_MODE, top_k=$TOP_K, keep_called=$KEEP_CALLED) | Endpoint check: $ENDPOINT_CHECK ($ENDPOINT_CHECK_MODE) | Mock responses: $ANALYSIS_MOCK_RESPONSES | Refine mode: $REFINE_MODE | Analyses: $ANALYSES"
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"
command -v ruff

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_GPU:-24}" \
    "$PYTHON" -m nbconvert --to notebook --execute refined.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$PROJECT_ROOT" \
    --output "slurm-${SLURM_JOB_NAME:-refined}-${SLURM_JOB_ID:-local}.ipynb"
