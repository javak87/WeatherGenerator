#!/bin/bash
# Build a flash-attn 2.8.3 wheel against the PyTorch 2.12 in this worktree.
# Run under screen so it keeps going after the terminal closes:
#   screen -dmS flash-attn-build ./build-flash-attn.sh
#   screen -r flash-attn-build
set -euo pipefail

ROOT=/e/project1/e-ext-2025e01-128/kasravi/zurich_hackathon
WORKTREE=$ROOT/develop-ssl-diffusion-performance-7fe82df1-nccl-2.27-symmetric-memory
WHEEL_DIR=$ROOT/wheels
LOG=$WORKTREE/build-flash-attn.log

export PATH="$HOME/.local/bin:$PATH"
source /e/software/default/lmod/lmod/init/bash
module load CUDA/12
# The venv interpreter is Python 3.12.3 from the Stages module. Without its
# lib directory, libpython resolves to /usr/lib64 and import torch segfaults.
export LD_LIBRARY_PATH="/e/software/default/stages/2025/software/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$WHEEL_DIR"
cd "$WORKTREE"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) start ====="
# shellcheck disable=SC1091
source .venv/bin/activate
# uv sync rewrites nvidia-cusparselt on this filesystem and the following
# setup.py then segfaults while loading it. Torch 2.12 is already installed.
python -c 'import torch; print("torch", torch.__version__)'
uv pip install pip setuptools wheel ninja packaging

export FLASH_ATTN_CUDA_ARCHS=90
export TORCH_CUDA_ARCH_LIST=9.0
export FLASH_ATTENTION_FORCE_BUILD=TRUE
# Eight nvcc processes at once run out of memory on this node and the
# compiler segfaults. Two is enough and the log stays readable.
export MAX_JOBS=2
export PYTHONUNBUFFERED=1

# pip wheel segfaults in its PEP 517 subprocess on this node. setup.py sees the
# installed torch and writes the wheel itself.
SRC=$ROOT/flash-attn-src
rm -rf "$SRC"
mkdir -p "$SRC"
curl -fsSL -o "$SRC/flash_attn-2.8.3.tar.gz" \
  https://files.pythonhosted.org/packages/source/f/flash-attn/flash_attn-2.8.3.tar.gz
tar -xzf "$SRC/flash_attn-2.8.3.tar.gz" -C "$SRC"
cd "$SRC/flash_attn-2.8.3"
pwd
python -u setup.py bdist_wheel
cp -v dist/*.whl "$WHEEL_DIR/"

echo "===== $(date) done ====="
ls -lh "$WHEEL_DIR"
