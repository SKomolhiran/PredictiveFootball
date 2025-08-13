# PredictiveFootball
My side project to create a predictive machine learning model to predict wins, draws, and losses and help me place good bets with an edge.

# Football Edge — Starter Repo

This is a minimal scaffold for your **pre‑match football prediction** project.

## Quickstart (VS Code + Conda)

1. **Open in VS Code** (File → Open Folder) and install extensions:
   - Python (ms-python.python), Jupyter, Pylance.
2. **Create the environment** (Terminal → New Terminal):
   ```bash
   conda env create -f environment.yml
   conda activate football-edge
   ```
3. **Select the interpreter**: VS Code Command Palette → *Python: Select Interpreter* → `football-edge`.
4. **Prepare folders & sanity check**:
   ```bash
   make prepare
   ```
5. **(Later) Train & backtest**:
   ```bash
   make train
   make backtest
   ```
6. **(Later) Run the Streamlit app**:
   ```bash
   make app
   ```

See `config/config.yaml` for basic settings.
