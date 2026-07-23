#!/bin/bash
# Submit the sepsis hyperparameter tuning (training/tune_sepsis.py) as SLURM jobs,
# one per reward mode. Each runs the full grid on the DEV set, selects by VAL FQE,
# and writes/merges tuned_configs.json. Run this BEFORE submit_all_jobs.sh (which
# consumes tuned_configs.json for the sepsis FINAL runs).
#
# Usage:
#   PARTITION=zen2_0256_a40x2 QOS=zen2_0256_a40x2 bash slurm/submit_tune_sepsis.sh
#   REWARDS="terminal" bash slurm/submit_tune_sepsis.sh   # subset
#
# NOTE: the two reward-mode tuning jobs both write tuned_configs.json (merging by
# reward key). To avoid a write race, they are chained: mixed waits for terminal.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

REWARDS="${REWARDS:-terminal mixed}"
N_STEPS="${N_STEPS:-50000}"
FQE_N_STEPS="${FQE_N_STEPS:-50000}"

SBATCH_OVERRIDE=()
[ -n "${PARTITION:-}" ] && SBATCH_OVERRIDE+=(--partition="$PARTITION")
[ -n "${QOS:-}" ] && SBATCH_OVERRIDE+=(--qos="$QOS")

PREV_JOB=""
for REWARD in $REWARDS; do
    DEP=()
    # chain: each tuning job waits for the previous, since they share tuned_configs.json
    [ -n "$PREV_JOB" ] && DEP=(--dependency=afterok:"$PREV_JOB")

    PY="/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_repro_py310/bin/python"
    DATA="${SEPSIS_DATA_DIR:-/gpfs/data/fs72297/jklotz/programming_data/sepsis_data}"
    JOB_ID=$(sbatch --parsable \
        --job-name="tune_$REWARD" \
        "${SBATCH_OVERRIDE[@]}" "${DEP[@]}" \
        --gres=gpu:1 --time=12:00:00 \
        --output="$REPO_DIR/logs/tune_${REWARD}_%j.out" \
        --error="$REPO_DIR/logs/tune_${REWARD}_%j.err" \
        --wrap="cd '$REPO_DIR' && SEPSIS_DATA_DIR='$DATA' '$PY' training/tune_sepsis.py \
                --device cuda:0 --reward_mode $REWARD --n_steps $N_STEPS --fqe_n_steps $FQE_N_STEPS")
    echo "Submitted tune_$REWARD: $JOB_ID${PREV_JOB:+  (after $PREV_JOB)}"
    PREV_JOB="$JOB_ID"
done

echo ""
echo "When both finish, tuned_configs.json will hold the best HP per algo/reward."
echo "Then: SKIP_SEPSIS=0 [PARTITION=... QOS=...] bash slurm/submit_all_jobs.sh"
