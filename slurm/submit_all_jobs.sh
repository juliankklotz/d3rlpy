#!/bin/bash
# Submit all benchmark and sepsis training jobs.
# 48 jobs total: CartPole (12) + Pong (12) + Sepsis-terminal (12) + Sepsis-mixed (12)
#
# Usage:
#   bash slurm/submit_all_jobs.sh
#
# Or dry-run (show commands without submitting):
#   DRY_RUN=1 bash slurm/submit_all_jobs.sh
#
# Sepsis jobs are gated behind SKIP_SEPSIS (default: skip) until FQE timing/cost
# is confirmed from the single TACR sanity-check job. To include them:
#   SKIP_SEPSIS=0 bash slurm/submit_all_jobs.sh
SKIP_SEPSIS="${SKIP_SEPSIS:-1}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO_DIR/slurm"
TEMPLATE="$SLURM_DIR/train_template.sh"
LOG_FILE="$REPO_DIR/slurm/submitted_jobs.log"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: train_template.sh not found at $TEMPLATE" >&2
    exit 1
fi

# cd into REPO_DIR: SLURM_SUBMIT_DIR (used by train_template.sh to locate
# itself, since ${BASH_SOURCE[0]} breaks under SLURM's spool copy) is set to
# the cwd at each sbatch call's submit time, not this script's own location.
cd "$REPO_DIR"

mkdir -p "$REPO_DIR/logs"
> "$LOG_FILE"

ALGOS=(discrete_bc discrete_cql discrete_dt discrete_tacr)
SEEDS=(0 1 2)

# Partition/QOS override: template defaults to A100 (often fully allocated).
# Set PARTITION/QOS to target A40 instead, e.g.
#   PARTITION=zen2_0256_a40x2 QOS=zen2_0256_a40x2 bash slurm/submit_all_jobs.sh
SBATCH_OVERRIDE=()
[ -n "${PARTITION:-}" ] && SBATCH_OVERRIDE+=(--partition="$PARTITION")
[ -n "${QOS:-}" ] && SBATCH_OVERRIDE+=(--qos="$QOS")
[ -n "${PARTITION:-}" ] && echo "Targeting partition: $PARTITION"

# Short codes so `squeue`'s 8-char-truncated NAME column stays distinguishable
declare -A ALGO_ABBR=(
    [discrete_bc]=bc
    [discrete_cql]=cql
    [discrete_dt]=dt
    [discrete_tacr]=tacr
)

count=0

# ── CartPole ──────────────────────────────────────────────────────────────────
echo "Submitting CartPole jobs..."
for ALGO in "${ALGOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${ALGO_ABBR[$ALGO]}_cp_s${SEED}"
        if [ -z "$DRY_RUN" ]; then
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "${SBATCH_OVERRIDE[@]}" "$TEMPLATE" "$ALGO" cartpole "$SEED" | awk '{print $NF}')
            echo "CartPole $ALGO seed$SEED ($JOB_NAME): $JOB_ID" | tee -a "$LOG_FILE"
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
        JOB_NAME="${ALGO_ABBR[$ALGO]}_pg_s${SEED}"
        if [ -z "$DRY_RUN" ]; then
            JOB_ID=$(sbatch --job-name="$JOB_NAME" "${SBATCH_OVERRIDE[@]}" "$TEMPLATE" "$ALGO" pong_minari "$SEED" | awk '{print $NF}')
            echo "Pong $ALGO seed$SEED ($JOB_NAME): $JOB_ID" | tee -a "$LOG_FILE"
        else
            echo "[DRY] sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO pong_minari $SEED"
        fi
        count=$((count + 1))
    done
done

if [ "$SKIP_SEPSIS" = "1" ]; then
    echo "Skipping Sepsis jobs (SKIP_SEPSIS=1, default). Set SKIP_SEPSIS=0 to include them."
else
    # ── Sepsis (terminal, fold 0 only — CV sweep not yet enabled) ─────────────
    echo "Submitting Sepsis (terminal reward) jobs..."
    for ALGO in "${ALGOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="${ALGO_ABBR[$ALGO]}_st_s${SEED}_f0"
            if [ -z "$DRY_RUN" ]; then
                JOB_ID=$(sbatch --job-name="$JOB_NAME" "${SBATCH_OVERRIDE[@]}" "$TEMPLATE" "$ALGO" sepsis "$SEED" 0 | awk '{print $NF}')
                echo "Sepsis-terminal $ALGO seed$SEED fold0 ($JOB_NAME): $JOB_ID" | tee -a "$LOG_FILE"
            else
                echo "[DRY] sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO sepsis $SEED 0"
            fi
            count=$((count + 1))
        done
    done

    # ── Sepsis (mixed, fold 0 only — CV sweep not yet enabled) ────────────────
    echo "Submitting Sepsis (mixed reward) jobs..."
    for ALGO in "${ALGOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="${ALGO_ABBR[$ALGO]}_sm_s${SEED}_f0"
            if [ -z "$DRY_RUN" ]; then
                export SLURM_ARGS="--reward_mode mixed"
                JOB_ID=$(sbatch --job-name="$JOB_NAME" "${SBATCH_OVERRIDE[@]}" "$TEMPLATE" "$ALGO" sepsis "$SEED" 0 | awk '{print $NF}')
                echo "Sepsis-mixed $ALGO seed$SEED fold0 ($JOB_NAME): $JOB_ID" | tee -a "$LOG_FILE"
                unset SLURM_ARGS
            else
                echo "[DRY] SLURM_ARGS='--reward_mode mixed' sbatch --job-name=$JOB_NAME $TEMPLATE $ALGO sepsis $SEED 0"
            fi
            count=$((count + 1))
        done
    done
fi

echo ""
if [ -z "$DRY_RUN" ]; then
    echo "✓ Submitted $count jobs. Log: $LOG_FILE"
    echo "Monitor: squeue -u jklotz"
else
    echo "DRY RUN: Would submit $count jobs."
    echo "Run without DRY_RUN=1 to actually submit."
fi
