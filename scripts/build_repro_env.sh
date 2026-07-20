#!/bin/bash
# Build a fresh, reproducible conda env from environment.yml WITHOUT touching
# the existing d3rlpy_dev_final_py310 env. Installs the local d3rlpy fork,
# then smoke-tests the Pong env recovery that the old env's ale-py 0.8.1 broke.
#
# Usage (on the cluster login node):
#   bash scripts/build_repro_env.sh
#
# After it prints ALL GREEN, update PYTHON= in slurm/train_template.sh and
# slurm/sanity_check.sh to the new env, then run slurm/sanity_check.sh.

set -euo pipefail

ENV_NAME="d3rlpy_repro_py310"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Building $ENV_NAME (old env left untouched) ==="
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "ERROR: env '$ENV_NAME' already exists. Remove it first with:" >&2
    echo "  conda env remove -n $ENV_NAME" >&2
    exit 1
fi

conda env create -n "$ENV_NAME" -f "$REPO_DIR/environment.yml"

echo "=== Installing local d3rlpy fork (editable) ==="
conda run -n "$ENV_NAME" pip install -e "$REPO_DIR" --no-deps

echo "=== Verifying versions ==="
conda run -n "$ENV_NAME" python -c "
import ale_py, gymnasium, minari, torch, numpy
print(f'ale-py      {ale_py.__version__}')
print(f'gymnasium   {gymnasium.__version__}')
print(f'minari      {minari.__version__}')
print(f'torch       {torch.__version__}')
print(f'numpy       {numpy.__version__}')
assert ale_py.__version__ >= '0.10', 'ale-py still too old!'
print('OK: ale-py is >=0.10 (compatible with gymnasium 1.0)')
"

echo "=== Smoke test: Pong env recovery (the thing that was broken) ==="
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
conda run -n "$ENV_NAME" python -c "
import d3rlpy
ds, env = d3rlpy.datasets.get_minari('atari/pong/expert-v0')
print('OK: Pong env recovered ->', type(env).__name__, 'obs', env.observation_space.shape)
" || {
    echo "" >&2
    echo "Pong env recovery still failing. If it is a missing ROM, run:" >&2
    echo "  conda run -n $ENV_NAME AutoROM --accept-license" >&2
    echo "then re-run this smoke test." >&2
    exit 1
}

echo ""
echo "=== ALL GREEN ==="
echo "Next: point slurm PYTHON= at:"
conda run -n "$ENV_NAME" python -c "import sys; print(sys.executable)"
echo "then run: bash slurm/sanity_check.sh"
