#!/bin/bash
# Build a fresh, reproducible conda env from environment.yml WITHOUT touching
# the existing d3rlpy_dev_final_py310 env.
#
# - Builds at an explicit --prefix on the DATA node (not ~/.conda/envs, which is
#   on the memory/quota-limited home fs).
# - Runs the heavy `conda env create` (torch+CUDA) on a COMPUTE node via SLURM:
#   pip/conda resolving multi-GB CUDA packages gets OOM-killed on the login node.
#
# Usage (on the cluster):
#   bash scripts/build_repro_env.sh              # submit build as a SLURM job
#   RUN_LOCAL=1 bash scripts/build_repro_env.sh  # run inline (needs an allocation with memory)
#
# After the build finishes and prints ALL GREEN, update PYTHON= in
# slurm/train_template.sh and slurm/sanity_check.sh to the printed new-env python.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Env prefix on the DATA node (same fs as the existing working env).
ENV_PREFIX="${ENV_PREFIX:-/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_repro_py310}"

# ── remove any stale/half-built env from a previous failed run ────────────────
if [ -d "$ENV_PREFIX" ]; then
    echo "Removing existing/half-built env at $ENV_PREFIX ..."
    conda env remove -p "$ENV_PREFIX" -y || rm -rf "$ENV_PREFIX"
fi

# ── the build steps, written to a standalone script the SLURM job runs ────────
BUILD_SCRIPT="$REPO_DIR/logs/_do_build_env.sh"
mkdir -p "$REPO_DIR/logs"
cat > "$BUILD_SCRIPT" <<BUILDEOF
#!/bin/bash
set -euo pipefail
# Source conda by absolute path — a fresh sbatch job may not have it on PATH.
CONDA_SH="/opt/sw/conda/miniconda3-24.1.2/etc/profile.d/conda.sh"
if [ -f "\$CONDA_SH" ]; then
    source "\$CONDA_SH"
else
    source "\$(conda info --base)/etc/profile.d/conda.sh"
fi

REPO_DIR="$REPO_DIR"
ENV_PREFIX="$ENV_PREFIX"

# Keep conda's package cache + tmp on the DATA node too (home is small).
export CONDA_PKGS_DIRS="/gpfs/data/fs72297/jklotz/.conda/pkgs"
export TMPDIR="/gpfs/data/fs72297/jklotz/tmp"
mkdir -p "\$CONDA_PKGS_DIRS" "\$TMPDIR"
# torch's cu128 wheel is multi-GB — stream it, don't buffer the whole thing
# (this + running on a compute node avoids the login-node OOM that killed the
# naive build).
export PIP_NO_CACHE_DIR=1

echo "=== conda env create at \$ENV_PREFIX (torch+CUDA via conda) ==="
conda env create -p "\$ENV_PREFIX" -f "\$REPO_DIR/environment.yml"

echo "=== Installing local d3rlpy fork (editable, no deps) ==="
conda run -p "\$ENV_PREFIX" pip install -e "\$REPO_DIR" --no-deps

echo "=== Verifying versions ==="
conda run -p "\$ENV_PREFIX" python -c "
import ale_py, gymnasium, minari, torch, numpy
print('ale-py     ', ale_py.__version__)
print('gymnasium  ', gymnasium.__version__)
print('minari     ', minari.__version__)
print('torch      ', torch.__version__)
print('numpy      ', numpy.__version__)
print('cuda avail ', torch.cuda.is_available())
assert ale_py.__version__ >= '0.10', 'ale-py still too old!'
assert torch.version.cuda is not None, 'torch has no CUDA build!'
print('OK: ale-py >=0.10 and torch has CUDA')
"

echo "=== Smoke test: Pong env recovery (the thing that was broken) ==="
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
conda run -p "\$ENV_PREFIX" python -c "
import d3rlpy
ds, env = d3rlpy.datasets.get_minari('atari/pong/expert-v0')
print('OK: Pong env recovered ->', type(env).__name__, 'obs', env.observation_space.shape)
" || {
    echo "Pong env recovery failed. If missing ROM, run:" >&2
    echo "  conda run -p \$ENV_PREFIX AutoROM --accept-license" >&2
    exit 1
}

echo ""
echo "=== ALL GREEN ==="
echo "New env python: \$ENV_PREFIX/bin/python"
BUILDEOF
chmod +x "$BUILD_SCRIPT"

if [ "${RUN_LOCAL:-0}" = "1" ]; then
    echo "Running build inline (RUN_LOCAL=1)..."
    bash "$BUILD_SCRIPT"
else
    echo "Submitting build to a compute node (avoids login-node OOM)..."
    JOB_ID=$(sbatch --parsable \
        --job-name=build_env \
        --partition=zen3_0512 \
        --qos=zen3_0512 \
        --mem=16G \
        --time=01:00:00 \
        --output="$REPO_DIR/logs/build_env_%j.out" \
        --error="$REPO_DIR/logs/build_env_%j.err" \
        "$BUILD_SCRIPT")
    echo "Submitted build job: $JOB_ID"
    echo "Watch:  tail -f $REPO_DIR/logs/build_env_${JOB_ID}.out"
    echo "Ready when it prints ALL GREEN."
fi
