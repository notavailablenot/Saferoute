# SafeRoute: Real-Time Traffic Sign Detection and Driver Warning System

CPE178P Foundations of AI, FOPI01, Group 01 (Avengoza, Hembrador, Ledda), Mapúa University.
Offline three-tier prototype: PyQt6 GUI (Presentation), FastAPI + ONNX Runtime + ByteTrack (Logic), SQLite (Data Access).

## Setup (WSL2 Ubuntu or Codespaces)
```bash
uv venv --python 3.12 --seed .venv   # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```
Datasets, model weights and the database are not committed (see `.gitignore`).