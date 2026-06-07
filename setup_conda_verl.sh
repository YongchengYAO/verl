#!/bin/bash
#
# Set up the `verl` conda environment for MedVision RFT by installing the frozen,
# known-good environment snapshot (requirements_medvision.txt). That snapshot pins the
# entire stack as pip wheels -- torch, vllm, sglang, flash-attn, flashinfer, and all
# CUDA libraries (nvidia-*-cu12) -- so no system CUDA/cuDNN install (and no root) is
# required. The only host prerequisite is an NVIDIA driver new enough for CUDA 12.8.
#
# Env vars:
#   ENV_NAME           conda env name (default: verl)
#   RUN_SYSTEM_FIXES   set to 1 (with sudo available) to apply the host GLIBC fix for
#                      flash-attn's `GLIBC_2.32' not found error (default: off)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-verl}"

# 1. Create + activate the conda env (idempotent)
eval "$(conda shell.bash hook)"
if [ ! -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    conda create -n "${ENV_NAME}" python=3.12 -y
else
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
fi
conda activate "${ENV_NAME}"
PYBIN="$CONDA_PREFIX/bin/python"

# 2. Prefer uv for fast, parallel installs; fall back to pip if it can't be bootstrapped.
#    uv targets the conda env explicitly via --python (conda envs are not venvs).
if ! command -v uv >/dev/null 2>&1; then
    pip install uv || true
fi
if command -v uv >/dev/null 2>&1; then
    INSTALL=(uv pip install --python "$PYBIN")
else
    INSTALL=(pip install)
fi

# 3. Install the full pinned environment from the frozen snapshot.
#    (torch / vllm / sglang / flash-attn / flashinfer + all CUDA libs as pip wheels)
#    --no-deps: the freeze is already a complete dependency closure, so we install the exact
#    pinned set without re-resolving. This is required because some packages declare overly
#    strict metadata bounds (e.g. datasets==3.6.0 pins multiprocess<0.70.17 / dill<0.3.9) that
#    contradict the known-good frozen versions and would otherwise make the resolver fail.
"${INSTALL[@]}" --no-deps -r "${SCRIPT_DIR}/requirements_medvision.txt"

# 4. Install verl itself (editable; not captured in the freeze).
"${INSTALL[@]}" --no-deps -e "${SCRIPT_DIR}"

# 5. (Optional) Host-level GLIBC fix for flash-attn `GLIBC_2.32' not found -- needs root.
#    Reference: https://github.com/modular/modular/issues/3684#issuecomment-2480409734
if [ "${RUN_SYSTEM_FIXES:-0}" = "1" ] && command -v sudo >/dev/null 2>&1; then
    echo "deb http://th.archive.ubuntu.com/ubuntu jammy main" | sudo tee -a /etc/apt/sources.list
    sudo apt-get update -y
    sudo apt-get install -y libc6 zlib1g-dev libtinfo-dev
else
    echo "Skipping system GLIBC fix (set RUN_SYSTEM_FIXES=1 with sudo if flash-attn reports GLIBC_2.32)."
fi

echo "Environment '${ENV_NAME}' is ready."
