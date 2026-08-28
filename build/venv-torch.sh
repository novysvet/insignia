#!/usr/bin/env bash
# Install CUDA torch into the oracle venv (runs INSIDE Arch WSL).
set -euo pipefail
V=/var/lib/insignia/oracle-venv
$V/bin/python -m pip --version >/dev/null 2>&1 || $V/bin/python -m ensurepip --upgrade >/dev/null
$V/bin/python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
$V/bin/python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
