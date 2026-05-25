# Thesis Prototype

## Setup

Requires Python 3.12.

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

Register the Jupyter kernel so VS Code can find it:

```bash
python -m ipykernel install --user --name thesis-venv --display-name "Python (thesis)"
```

Then open `prototype.ipynb` in VS Code and select the **Python (thesis)** kernel.

## Environment variables

Copy `.env.example` to `.env` and fill in your API key:

```
NEBIUS_API_KEY=your_key_here
```
