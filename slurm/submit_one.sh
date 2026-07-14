#!/bin/bash
# Submit a single training job, ensuring logs/ exists first.
# (SLURM resolves #SBATCH --output=logs/... at submit time, before the job
#  script body runs — mkdir -p logs inside the job script is too late.)
#
# Usage:
#   slurm/submit_one.sh ALGO DATASET SEED [FOLD] [-- extra python args]
#
# Examples:
#   slurm/submit_one.sh discrete_tacr sepsis 0 0 -- --reward_mode mixed
#   slurm/submit_one.sh discrete_bc cartpole 0

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"

ALGO="${1:?Usage: submit_one.sh ALGO DATASET SEED [FOLD] [-- extra python args]}"
DATASET="${2:?}"
SEED="${3:?}"
FOLD="${4:-0}"
shift 4 2>/dev/null || shift $#

# Anything after '--' is forwarded to the python script via SLURM_ARGS
if [ "${1:-}" = "--" ]; then
    shift
    export SLURM_ARGS="$*"
fi

# Short codes so `squeue`'s 8-char-truncated NAME column stays distinguishable
declare -A ALGO_ABBR=([discrete_bc]=bc [discrete_cql]=cql [discrete_dt]=dt [discrete_tacr]=tacr)
declare -A DATASET_ABBR=([cartpole]=cp [pong_minari]=pg [pong]=pg [sepsis]=se)
ALGO_SHORT="${ALGO_ABBR[$ALGO]:-$ALGO}"
DATASET_SHORT="${DATASET_ABBR[$DATASET]:-$DATASET}"

JOB_NAME="${ALGO_SHORT}_${DATASET_SHORT}_s${SEED}"
[ "$DATASET" = "sepsis" ] && JOB_NAME="${JOB_NAME}_f${FOLD}"

JOB_ID=$(sbatch --job-name="$JOB_NAME" "$REPO_DIR/slurm/train_template.sh" "$ALGO" "$DATASET" "$SEED" "$FOLD" | awk '{print $NF}')

echo "Submitted $JOB_NAME: $JOB_ID"
