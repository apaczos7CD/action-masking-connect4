#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
VENV_DIR=".venv"

echo "Checking Python version..."
$PYTHON_BIN --version

echo "Creating virtual environment if needed..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing exact dependencies..."
pip install -r requirements_exact.txt

echo "Creating output directories..."
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

echo "Training Models..."
python -m scripts.train --config "configs/train_ppo.yaml"
python -m scripts.train --config "configs/train_maskable_ppo.yaml"

echo "Evaluating checkpoints..."
python -m scripts.eval --config "configs/eval.yaml"

echo "Summarize..."
python -m scripts.summarize --config "configs/summarize.yaml"

echo "Generating plots..."
python -m scripts.plot --config "configs/plot.yaml"
python -m scripts.plot --config "configs/plot_avg.yaml"

echo "Reproduction finished successfully."