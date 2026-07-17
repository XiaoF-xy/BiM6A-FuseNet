#!/usr/bin/env bash
set -euo pipefail

# Use this if the server already has python3.10 but does not use Conda.
# It creates the virtual environment inside BiM6A-FuseNet/.venv-bim6a.
#
# Usage:
#   cd BiM6A-FuseNet
#   bash env/create_venv_cuda121.sh
#   source .venv-birna/bin/activate
#   python env/check_runtime.py

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-bim6a}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r env/requirements_common.txt

python env/check_runtime.py
