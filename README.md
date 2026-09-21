# Thesis Prototype

## Setup

Python 3.9 or newer is supported. Python 3.12 is recommended locally; the
cluster environment currently provides Python 3.9.

```bash
python3.12 -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On the cluster, first check whether the cluster provides a PyTorch/CUDA module:

```bash
module avail pytorch cuda
```

Use the documented cluster PyTorch module or wheel before installing the
requirements. Do not install a second incompatible PyTorch build into the
environment. Verify the GPU from inside a Slurm job, not on the login node:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Register the Jupyter kernel so VS Code can find it:

```bash
python -m ipykernel install --user --name thesis-venv --display-name "Python (thesis)"
```

Then open `prototype.ipynb` in VS Code and select the **Python (thesis)** kernel.

## Local model backend

The prototype now loads a Hugging Face causal language model in-process. The public
functions in `llm.py` are unchanged, so the notebooks and endpoint-check refinement
path can use the local backend without an API token.

Set the model and generation options before starting Python:

```bash
export LOCAL_MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct
export LOCAL_MAX_NEW_TOKENS=1024
export LOCAL_DO_SAMPLE=0
```

The first run downloads the model into the Hugging Face cache. On a cluster where
compute nodes cannot access Hugging Face, download it to a shared cache first and
run with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/path/to/shared/huggingface-cache
export HF_HUB_OFFLINE=1
```

Run a local smoke test after installing the requirements:

```bash
python local_smoke_test.py --model "$LOCAL_MODEL_ID"
```

## Slurm

The initial launcher targets the `dev_gpu_h100` partition with one GPU, 24 CPUs,
180 GB host memory, and 30 minutes:

```bash
sbatch slurm/local_smoke_test.sbatch
```

Override the model or cache at submission time when needed:

```bash
LOCAL_MODEL_ID=Qwen/Qwen2.5-Coder-14B-Instruct \
HF_HOME=/shared/$USER/huggingface \
sbatch slurm/local_smoke_test.sbatch
```

The smoke job is intentionally separate from the full benchmark. Once model load,
CUDA, output format, and latency are confirmed, use the same environment variables
when executing the baseline or refined notebook under Slurm.
