#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --job-name=restgpt-local-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

MODEL_ID="${LOCAL_MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
export LOCAL_MODEL_ID="$MODEL_ID"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export LOCAL_MAX_NEW_TOKENS="${LOCAL_MAX_NEW_TOKENS:-256}"
export LOCAL_DO_SAMPLE="${LOCAL_DO_SAMPLE:-0}"

if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
fi

PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Model: $LOCAL_MODEL_ID"
echo "HF_HOME: $HF_HOME"
nvidia-smi
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available()"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_GPU:-24}" \
    "$PYTHON" local_smoke_test.py --model "$LOCAL_MODEL_ID"