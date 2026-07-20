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
PYTHON="/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_repro_py310/bin/python"
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
# Resolve REPO_DIR by trying candidates in order and picking the first that
# actually contains the repo. Env-var sniffing alone is unreliable:
# - under sbatch, SLURM copies this script to a spool dir, so ${BASH_SOURCE[0]}
#   resolves to /var/spool/slurm/slurmd/scripts/... not the repo
# - under an interactive JupyterHub session, SLURM_SUBMIT_DIR is pre-set to a
#   bogus /opt/jupyterhub (and SLURM_JOB_ID may also be set, so it cannot be
#   used to discriminate)
# SLURM_SUBMIT_DIR is tried first since that is the correct source for real
# batch jobs; each candidate is validated before being accepted.
REPO_DIR=""
for _candidate in \
    "${SLURM_SUBMIT_DIR:-}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
    "$PWD"
do
    if [ -n "$_candidate" ] && [ -f "$_candidate/scripts/cluster_validate" ]; then
        REPO_DIR="$_candidate"
        break
    fi
done

if [ -z "$REPO_DIR" ]; then
    echo "FATAL: could not locate repo root (no candidate contained scripts/cluster_validate)." >&2
    echo "       BASH_SOURCE=${BASH_SOURCE[0]}" >&2
    echo "       SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    echo "       PWD=$PWD" >&2
    exit 1
fi
echo "REPO_DIR resolved to: $REPO_DIR"

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
    cartpole|pong|pong_minari)
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
