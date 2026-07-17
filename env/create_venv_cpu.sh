#!/usr/bin/env bash
set -euo pipefail

# CPU-only fallback. This is useful for debugging the code path, not for real training.
#
# Usage:
#   cd BiM6A-FuseNet
#   bash env/create_venv_cpu.sh
#   source .venv-birna-cpu/bin/activate
#   python env/check_runtime.py

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-bim6a-cpu}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r env/requirements_common.txt

python env/check_runtime.py
