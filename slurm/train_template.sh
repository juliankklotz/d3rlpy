#!/bin/bash
#SBATCH --job-name={ALGO}_{DATASET}
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/{ALGO}_{DATASET}_seed{SEED}_%j.out
#SBATCH --error=logs/{ALGO}_{DATASET}_seed{SEED}_%j.err

# ── data paths ────────────────────────────────────────────────────────────────
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export SEPSIS_DATA_DIR="/gpfs/data/fs72297/jklotz/programming_data/sepsis_data"

# ── locate python (works with both personal and shared conda) ─────────────────
ENV_NAME="d3rlpy_dev_final_py310"
ENV_PATH="$(conda env list 2>/dev/null | grep -E "^${ENV_NAME}\s" | awk '{print $NF}')"
if [ -z "$ENV_PATH" ] || [ ! -f "$ENV_PATH/bin/python" ]; then
    echo "ERROR: conda env '$ENV_NAME' not found. Run scripts/cluster_env_setup first." >&2
    exit 1
fi
PYTHON="$ENV_PATH/bin/python"

spackup cuda-zen

# ── pre-flight validation ────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PYTHON" bash "$REPO_DIR/scripts/cluster_validate" || {
    echo "FATAL: environment validation failed — aborting job." >&2
    echo "Fix: bash $REPO_DIR/scripts/cluster_env_setup --rebuild" >&2
    exit 1
}

# ── training ──────────────────────────────────────────────────────────────────
mkdir -p logs
echo "Starting: algo={ALGO}  dataset={DATASET}  seed={SEED}"
echo "Python:   $PYTHON  ($(hostname))"

"$PYTHON" "$REPO_DIR/training/train.py" \
    --algo {ALGO} \
    --dataset {DATASET} \
    --seed {SEED} \
    --device cuda:0

echo "Done."
