#!/bin/bash
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only job: no --gres=gpu, no A100 partition — sits in a different,
# much faster queue than the training jobs. Runs the exact same path
# resolution + environment validation as train_template.sh, without
# actually training anything. Use this to confirm SLURM plumbing (REPO_DIR
# resolution, python env, data paths, imports) works BEFORE submitting to
# the A100 queue, where a bad config costs a ~24h wait per retry.
#
# Usage:
#   slurm/submit_sanity_check.sh
# or directly:
#   sbatch slurm/sanity_check.sh

set -eu

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

echo "=== SLURM sanity check ==="
echo "Host:            $(hostname)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_DIR resolved to: $REPO_DIR"
echo ""

# ── hardcoded python path (must match train_template.sh) ─────────────────────
PYTHON="/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_dev_final_py310/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "FAIL: python not found at $PYTHON" >&2
    echo "Run: conda env list  # to find current path" >&2
    exit 1
fi
echo "OK: python found at $PYTHON"

# ── data paths (must match train_template.sh) ─────────────────────────────────
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"
export SEPSIS_DATA_DIR="/gpfs/data/fs72297/jklotz/programming_data/sepsis_data"

for f in mimic_dataset.csv sepsis_cohort.csv; do
    if [ -f "$SEPSIS_DATA_DIR/$f" ]; then
        echo "OK: found $SEPSIS_DATA_DIR/$f"
    else
        echo "FAIL: missing $SEPSIS_DATA_DIR/$f" >&2
        exit 1
    fi
done

# ── same validation train_template.sh runs before every real job ─────────────
if [ ! -f "$REPO_DIR/scripts/cluster_validate" ]; then
    echo "FAIL: REPO_DIR resolved to '$REPO_DIR' but scripts/cluster_validate not found there." >&2
    exit 1
fi
echo "OK: scripts/cluster_validate found at correct path"

PYTHON="$PYTHON" bash "$REPO_DIR/scripts/cluster_validate" || {
    echo "FAIL: environment validation failed." >&2
    exit 1
}

# ── confirm the actual training entry points exist and are importable ────────
"$PYTHON" - <<PYEOF
import sys
sys.path.insert(0, "$REPO_DIR")
import d3rlpy
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_fold
print("OK: d3rlpy + FQE + sepsis_loader imports succeed")
PYEOF

echo ""
echo "=== ALL CHECKS PASSED ==="
