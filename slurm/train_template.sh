#!/bin/bash
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_seed%A_%j.out
#SBATCH --error=logs/%x_seed%A_%j.err

# Positional args (passed via `sbatch train_template.sh ALGO DATASET SEED [FOLD]`):
#   $1 = ALGO      (discrete_bc, discrete_cql, discrete_dt, discrete_tacr)
#   $2 = DATASET   (cartpole, pong, sepsis)
#   $3 = SEED      (0, 1, 2, ...)
#   $4 = FOLD      (0-4, sepsis only, default 0)
ALGO="$1"
DATASET="$2"
SEED="$3"
FOLD="${4:-0}"

if [ -z "$ALGO" ] || [ -z "$DATASET" ] || [ -z "$SEED" ]; then
    echo "ERROR: usage: sbatch train_template.sh ALGO DATASET SEED [FOLD]" >&2
    exit 1
fi

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
# NOTE: SLURM copies the submitted script to a spool dir before execution, so
# ${BASH_SOURCE[0]} does NOT point at the real repo checkout here — it resolves
# to something like /var/spool/slurm/slurmd/scripts/... Use SLURM_SUBMIT_DIR
# (which sbatch sets to the cwd at submission time) instead. Only trust it
# inside a real SLURM job (SLURM_JOB_ID set): an interactive JupyterHub session
# pre-exports a bogus SLURM_SUBMIT_DIR=/opt/jupyterhub, so a `bash train_template.sh`
# there must self-locate via BASH_SOURCE instead.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [ ! -f "$REPO_DIR/scripts/cluster_validate" ]; then
    echo "FATAL: REPO_DIR resolved to '$REPO_DIR' but scripts/cluster_validate not found there." >&2
    echo "       SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi

PYTHON="$PYTHON" bash "$REPO_DIR/scripts/cluster_validate" || {
    echo "FATAL: environment validation failed — aborting job." >&2
    exit 1
}

# ── training ──────────────────────────────────────────────────────────────────
mkdir -p logs
echo "Starting: algo=$ALGO  dataset=$DATASET  seed=$SEED  fold=$FOLD"
echo "Python:   $PYTHON  ($(hostname))"
echo "Args:     ${SLURM_ARGS:-none}"

case "$DATASET" in
    cartpole|pong)
        SCRIPT="$REPO_DIR/training/train_benchmarks.py"
        EXTRA_ARGS="--dataset $DATASET"
        ;;
    sepsis)
        SCRIPT="$REPO_DIR/training/train_sepsis.py"
        EXTRA_ARGS="--fold $FOLD"
        ;;
    *)
        echo "ERROR: unknown dataset $DATASET" >&2
        exit 1
        ;;
esac

"$PYTHON" "$SCRIPT" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --device cuda:0 \
    $EXTRA_ARGS \
    ${SLURM_ARGS:-}

echo "Done."
