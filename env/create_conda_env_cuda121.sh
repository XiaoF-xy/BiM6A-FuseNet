#!/usr/bin/env bash
set -euo pipefail

# Recommended for Linux cloud servers with NVIDIA driver support for CUDA 12.1.
# Python version: 3.10
# Environment name: bim6a_fusenet
#
# Usage:
#   cd BiM6A-FuseNet
#   bash env/create_conda_env_cuda121.sh
#   conda activate birna_m6a
#   python env/check_runtime.py

ENV_NAME="${ENV_NAME:-bim6a_fusenet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

# Official PyTorch wheel index pattern for CUDA builds.
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r env/requirements_common.txt

python env/check_runtime.py
