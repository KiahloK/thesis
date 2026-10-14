#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --job-name=baseline
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Runs baseline.ipynb headless: one plain LLM call per query over the full, unfiltered spec,
# then evaluation. Results land in output/<date>_baseline/; the executed notebook (with all
# printed output) is saved as slurm-baseline-<jobid>.ipynb.
#
# Every notebook setting can be overridden at submission time. To split a baseline across
# dev-queue jobs, run chunks with QUERY_OFFSET=0, 10, 20, ... and QUERY_LIMIT=10, then refine
# all chunks together with refined.sbatch: LOAD_BASELINE=1 BASELINE_DIR=output/<chunk1>,output/<chunk2>,...

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
# The baseline is the plain generation call; reasoning stays off unless set explicitly
# (see llm.py / refined.sbatch for the levels).
export LOCAL_REASONING_GENERATE="${LOCAL_REASONING_GENERATE:-${LOCAL_REASONING:-off}}"
export LOCAL_MAX_WORKERS="${LOCAL_MAX_WORKERS:-1}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

# Notebook configuration (see the Configuration cell in baseline.ipynb).
export BENCHMARK_TYPES="${BENCHMARK_TYPES:-socbenchd_1}"
export QUERY_LIMIT="${QUERY_LIMIT:-10}"
export QUERY_OFFSET="${QUERY_OFFSET:-0}"

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi

if [[ ! -d benchmark/"${BENCHMARK_TYPES%%,*}" ]]; then
    echo "Benchmark data not found under $PROJECT_ROOT/benchmark (it is gitignored - copy it over)." >&2
    exit 1
fi

echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Model: $LOCAL_MODEL_ID | HF_HOME: $HF_HOME | HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}"
echo "Reasoning: generate=$LOCAL_REASONING_GENERATE"
echo "Benchmarks: $BENCHMARK_TYPES | Queries: $QUERY_LIMIT (offset $QUERY_OFFSET)"
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_GPU:-24}" \
    "$PYTHON" -m nbconvert --to notebook --execute baseline.ipynb \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir "$PROJECT_ROOT" \
    --output "slurm-${SLURM_JOB_NAME:-baseline}-${SLURM_JOB_ID:-local}.ipynb"
