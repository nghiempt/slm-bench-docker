#!/bin/bash
set -e

# Create and activate virtualenv
python3 -m venv venv_m4
source venv_m4/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install datasets transformers scikit-learn wandb tqdm

python3 run.py
