#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${1:-/bin/python3}"

"${python_bin}" -m pip install --user -r "${repo_dir}/requirements-rti.txt"
mkdir -p "${repo_dir}/.third_party"
if [[ ! -d "${repo_dir}/.third_party/cusadi/.git" ]]; then
    git clone https://github.com/se-hwan/cusadi.git "${repo_dir}/.third_party/cusadi"
fi
git -C "${repo_dir}/.third_party/cusadi" checkout bffea25849a901277a8f8dc4db16748e629f9053
"${python_bin}" -c 'import torch, casadi, osqp, importlib.metadata as metadata; print("CUDA:", torch.cuda.is_available(), "CasADi:", casadi.__version__, "OSQP:", osqp.__version__, "cuDSS:", metadata.version("nvidia-cudss-cu13"))'
