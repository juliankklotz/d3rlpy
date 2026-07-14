#!/bin/bash
# Submit all benchmark and sepsis training jobs.
# 48 jobs total: CartPole (12) + Pong (12) + Sepsis-terminal (12) + Sepsis-mixed (12)
#
# Usage:
#   bash slurm/submit_all_jobs.sh
#
# Or dry-run (show commands without submitting):
#   DRY_RUN=1 bash slurm/submit_all_jobs.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO_DIR/slurm"
TEMPLATE="$SLURM_DIR/train_template.sh"
LOG_FILE="$REPO_DIR/slurm/submitted_jobs.log"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: train_template.sh not found at $TEMPLATE" >&2
    exit 1
fi

mkdir -p "$REPO_DIR/logs"
> "$LOG_FILE"

ALGOS=(discrete_bc discrete_cql discrete_dt discrete_tacr)
SEEDS=(0 1 2)

count=0

# ── CartPole ──────────────────────────────────────────────────────────────────
echo "Submitting CartPole jobs..."
for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${ALGO}_cartpole_seed${SEED}"
        if [ -z "$DRY_RUN" ]; then
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "$TEMPLATE" "$ALGO" cartpole "$SEED" | awk '{print $NF}')
            echo "CartPole $ALGO seed$SEED: $JOB_ID" | tee -a "$LOG_FILE"
        else
            echo "[DRY] sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO cartpole $SEED"
        fi
        count=$((count + 1))
    done
done

# ── Pong (minari) ─────────────────────────────────────────────────────────────
echo "Submitting Pong (minari) jobs..."
for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${ALGO}_pong_seed${SEED}"
        if [ -z "$DRY_RUN" ]; then
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "$TEMPLATE" "$ALGO" pong_minari "$SEED" | awk '{print $NF}')
            echo "Pong $ALGO seed$SEED: $JOB_ID" | tee -a "$LOG_FILE"
        else
            echo "[DRY] sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO pong_minari $SEED"
        fi
        count=$((count + 1))
    done
done

# ── Sepsis (terminal, fold 0 only — CV sweep not yet enabled) ─────────────────
echo "Submitting Sepsis (terminal reward) jobs..."
for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${ALGO}_sepsis_terminal_seed${SEED}_fold0"
        if [ -z "$DRY_RUN" ]; then
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "$TEMPLATE" "$ALGO" sepsis "$SEED" 0 | awk '{print $NF}')
            echo "Sepsis-terminal $ALGO seed$SEED fold0: $JOB_ID" | tee -a "$LOG_FILE"
        else
            echo "[DRY] sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO sepsis $SEED 0"
        fi
        count=$((count + 1))
    done
done

# ── Sepsis (mixed, fold 0 only — CV sweep not yet enabled) ────────────────────
echo "Submitting Sepsis (mixed reward) jobs..."
for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${ALGO}_sepsis_mixed_seed${SEED}_fold0"
        if [ -z "$DRY_RUN" ]; then
            export SLURM_ARGS="--reward_mode mixed"
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "$TEMPLATE" "$ALGO" sepsis "$SEED" 0 | awk '{print $NF}')
            echo "Sepsis-mixed $ALGO seed$SEED fold0: $JOB_ID" | tee -a "$LOG_FILE"
            unset SLURM_ARGS
        else
            echo "[DRY] SLURM_ARGS='--reward_mode mixed' sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO sepsis $SEED 0"
        fi
        count=$((count + 1))
    done
done

echo ""
if [ -z "$DRY_RUN" ]; then
    echo "✓ Submitted $count jobs. Log: $LOG_FILE"
    echo "Monitor: squeue -u jklotz"
else
    echo "DRY RUN: Would submit $count jobs."
    echo "Run without DRY_RUN=1 to actually submit."
fi
