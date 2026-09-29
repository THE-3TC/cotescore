#!/usr/bin/env bash
#
# setup_lightning_gpu.sh
# ----------------------
# Idempotent environment setup for running the PP-DocLayout / cotescore
# benchmarks on a Lightning AI studio with GPU (CUDA 13).
#
# Why this exists (the painful lessons baked in):
#   * Lightning's `/system/conda` partition is tiny (~9 GB) and uv's cache
#     defaults to live there, so installs die with "No space left on device"
#     even though the studio disk has hundreds of GB free. We pin UV_CACHE_DIR
#     to the project disk (same filesystem as the venv, so uv can hardlink).
#   * PaddlePaddle GPU wheels for cu130 are CPython 3.12 only -> the venv MUST
#     be Python 3.12, not 3.13, or paddle silently falls back to CPU.
#   * DocLayout-YOLO and Docling Heron run on torch, so torch must be the
#     CUDA (cu130) build - the CPU wheel makes both fail under --device cuda.
#   * Running PP-DocLayout on CPU hits a Paddle PIR/oneDNN crash
#     (ConvertPirAttribute2RuntimeAttribute). GPU avoids it entirely; this
#     script's whole point is to make the GPU path actually work.
#
# Usage:
#   bash scripts/setup_lightning_gpu.sh          # set up / verify
#   source scripts/setup_lightning_gpu.sh        # also leaves the venv active
#
# Re-running is cheap: it skips work that's already done.

set -euo pipefail

# --- Configuration ----------------------------------------------------------
PYTHON_VERSION="3.12"
PADDLE_VERSION="3.3.0"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu130/"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"
VENV_DIR=".venv-gpu"

# Resolve the project root from this script's location so the script works
# regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Keep uv's cache on the project disk (NOT /system/conda) and on the same
# filesystem as the venv so uv can hardlink instead of copying.
export UV_CACHE_DIR="${PROJECT_ROOT}/.uv-cache"

echo "=============================================================="
echo " Lightning GPU setup for: ${PROJECT_ROOT}"
echo " UV_CACHE_DIR = ${UV_CACHE_DIR}"
echo "=============================================================="

# --- Sanity: uv must be installed -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

mkdir -p "${UV_CACHE_DIR}"

# --- Create the venv (Python 3.12) if missing -------------------------------
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[1/5] Creating Python ${PYTHON_VERSION} venv at ${VENV_DIR}..."
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
    echo "[1/5] venv ${VENV_DIR} already exists - reusing."
fi

# Use the venv's interpreter explicitly so we don't depend on activation state.
VPY="${PROJECT_ROOT}/${VENV_DIR}/bin/python"

cuda_ok() {
    "${VPY}" - <<'PY' 2>/dev/null
import sys
try:
    import paddle
    sys.exit(0 if paddle.is_compiled_with_cuda() else 1)
except Exception:
    sys.exit(1)
PY
}

# --- Install GPU PaddlePaddle (cu130, cp312) --------------------------------
if cuda_ok; then
    echo "[2/5] paddlepaddle-gpu already CUDA-enabled - skipping."
else
    echo "[2/5] Installing paddlepaddle-gpu==${PADDLE_VERSION} (cu130)..."
    uv pip install --python "${VPY}" \
        "paddlepaddle-gpu==${PADDLE_VERSION}" -i "${PADDLE_INDEX}"
fi

# --- Verify CUDA before going further ---------------------------------------
echo "[3/5] Verifying Paddle sees the GPU..."
if cuda_ok; then
    echo "      OK: paddle.is_compiled_with_cuda() == True"
else
    echo "ERROR: Paddle is NOT CUDA-enabled after install." >&2
    echo "       Check that ${VENV_DIR} is Python 3.12 (cu130 wheels are cp312-only)" >&2
    echo "       and that the cu130 GPU wheel actually downloaded (~1.36 GB)." >&2
    exit 1
fi

# --- Install CUDA torch + the project + paddleocr ---------------------------
# torch runs DocLayout-YOLO and Heron on the GPU. Install it before the project
# so resolving ".[benchmarks]" keeps the cu130 build instead of pulling CPU torch.
echo "[4/5] Installing CUDA torch, project (.[benchmarks]) and paddleocr..."
uv pip install --python "${VPY}" \
    torch torchvision --index-url "${TORCH_CUDA_INDEX}"
uv pip install --python "${VPY}" -e ".[benchmarks]" paddleocr

# --- Verify torch sees the GPU ----------------------------------------------
echo "[5/5] Verifying torch sees the GPU..."
if "${VPY}" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "      OK: torch.cuda.is_available() == True"
else
    echo "ERROR: torch cannot see the GPU; YOLO and Heron would fail on --device cuda." >&2
    echo "       Check that the cu130 torch wheel installed (not the CPU build)." >&2
    exit 1
fi

echo "=============================================================="
echo " Setup complete."
echo " Activate with:  source ${VENV_DIR}/bin/activate"
echo " Then run, e.g.:"
echo "   python scripts/benchmark_hiertext.py \\"
echo "       --dataset ~/validation \\"
echo "       --groundtruth ~/hiertext/gt/validation.jsonl \\"
echo "       --device cuda --models yolo heron ppdoc-l"
echo "=============================================================="

# If this script was sourced (not executed), leave the venv active for the
# caller's interactive shell.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "venv activated in current shell."
fi
