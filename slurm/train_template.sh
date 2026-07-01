#!/bin/bash
#SBATCH --job-name={ALGO}_{DATASET}
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/{ALGO}_{DATASET}_seed{SEED}_%j.out
#SBATCH --error=logs/{ALGO}_{DATASET}_seed{SEED}_%j.err

# ── environment ───────────────────────────────────────────────────────────────
ENV_NAME="d3rlpy_dev_final_py310"
CONDA_BASE="${CONDA_PREFIX:-$HOME/miniconda3}"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"

export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export SEPSIS_DATA_DIR="/gpfs/data/fs72297/jklotz/programming_data/sepsis_data"

spackup cuda-zen

# ── pre-flight validation ────────────────────────────────────────────────────
# Fails the job immediately if env is broken — no wasted GPU time.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PYTHON" bash "$REPO_DIR/scripts/cluster_validate" || {
    echo "FATAL: environment validation failed — aborting job"
    exit 1
}

# ── training ──────────────────────────────────────────────────────────────────
mkdir -p logs

echo "Starting: algo={ALGO}  dataset={DATASET}  seed={SEED}"
echo "Python:   $PYTHON"
echo "Node:     $(hostname)"

"$PYTHON" "$REPO_DIR/training/train.py" \
    --algo {ALGO} \
    --dataset {DATASET} \
    --seed {SEED} \
    --device cuda:0

echo "Done."
