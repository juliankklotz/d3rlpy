#!/bin/bash
#SBATCH --job-name={ALGO}_{DATASET}
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/{ALGO}_{DATASET}_seed{SEED}_%j.out
#SBATCH --error=logs/{ALGO}_{DATASET}_seed{SEED}_%j.err

# ── hardcoded python path (update if env recreated) ──────────────────────────
PYTHON="/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_dev_final_py310/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: python not found at $PYTHON" >&2
    echo "Run: conda env list  # to find current path" >&2
    exit 1
fi

# ── data paths ────────────────────────────────────────────────────────────────
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export SEPSIS_DATA_DIR="/gpfs/data/fs72297/jklotz/programming_data/sepsis_data"

# ── pre-flight validation ────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PYTHON" bash "$REPO_DIR/scripts/cluster_validate" || {
    echo "FATAL: environment validation failed — aborting job." >&2
    exit 1
}

# ── training ──────────────────────────────────────────────────────────────────
mkdir -p logs
echo "Starting: algo={ALGO}  dataset={DATASET}  seed={SEED}"
echo "Python:   $PYTHON  ($(hostname))"
echo "Args:     ${SLURM_ARGS:-none}"

case "{DATASET}" in
    cartpole|pong)
        SCRIPT="$REPO_DIR/training/train_benchmarks.py"
        ;;
    sepsis)
        SCRIPT="$REPO_DIR/training/train_sepsis.py"
        ;;
    *)
        echo "ERROR: unknown dataset {DATASET}" >&2
        exit 1
        ;;
esac

"$PYTHON" "$SCRIPT" \
    --algo {ALGO} \
    --dataset {DATASET} \
    --seed {SEED} \
    --device cuda:0 \
    ${SLURM_ARGS:-}

echo "Done."
