#!/bin/bash
# Submit the CPU-only sanity check (slurm/sanity_check.sh).
# Fast queue, no GPU, no A100 partition — use before submitting real training
# jobs to catch config/path/env problems in minutes instead of after a ~24h
# A100 queue wait.
#
# Usage:
#   slurm/submit_sanity_check.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_DIR/logs"

JOB_ID=$(cd "$REPO_DIR" && sbatch --job-name=sanity_check "$REPO_DIR/slurm/sanity_check.sh" | awk '{print $NF}')

echo "Submitted sanity_check: $JOB_ID"
echo "Check with: squeue -u \$USER | grep sanity"
echo "Log at:     logs/sanity_check_${JOB_ID}.out"
