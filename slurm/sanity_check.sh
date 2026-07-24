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

# Resolve REPO_DIR by trying candidates in order and picking the first that
# actually contains the repo. Env-var sniffing alone is unreliable here:
# - under sbatch, ${BASH_SOURCE[0]} points at SLURM's spool copy, not the repo
# - under an interactive JupyterHub session on a compute node, SLURM_SUBMIT_DIR
#   is pre-set to a bogus /opt/jupyterhub (and SLURM_JOB_ID may also be set,
#   so it cannot be used to discriminate)
# Validating each candidate sidesteps the guessing entirely.
REPO_DIR=""
for _candidate in \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
    "${SLURM_SUBMIT_DIR:-}" \
    "$PWD"
do
    if [ -n "$_candidate" ] && [ -f "$_candidate/scripts/cluster_validate" ]; then
        REPO_DIR="$_candidate"
        break
    fi
done

if [ -z "$REPO_DIR" ]; then
    echo "FAIL: could not locate repo root (no candidate contained scripts/cluster_validate)." >&2
    echo "      BASH_SOURCE=${BASH_SOURCE[0]}" >&2
    echo "      SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    echo "      PWD=$PWD" >&2
    echo "      Run this from the repo root: cd <repo> && bash slurm/sanity_check.sh" >&2
    exit 1
fi

echo "=== SLURM sanity check ==="
echo "Host:            $(hostname)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "REPO_DIR resolved to: $REPO_DIR"
echo ""

# ── hardcoded python path (must match train_template.sh) ─────────────────────
PYTHON="/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_repro_py310/bin/python"
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
from d3rlpy.sepsis_loader import get_sepsis_splits, get_sepsis_dev_buffer
print("OK: d3rlpy + FQE + sepsis_loader imports succeed")
PYEOF

# ── smoke-run the REAL training entry points (tiny steps, CPU) ────────────────
# This drives train_sepsis.py / train_benchmarks.py exactly as the batch jobs
# do, on real data, through the full pipeline (load -> build -> fit -> eval/FQE),
# just with 5 steps instead of 100k. Catches data/column/scaler/dispatch bugs
# that imports alone miss — the class of bug that has repeatedly reached the
# A100 queue. Skip with SKIP_SMOKE=1 for a fast import-only check.
# Device for the smoke runs. Default cpu (runs anywhere, no GPU allocation).
# Set SMOKE_DEVICE=cuda:0 to exercise the real GPU path in an interactive A100
# session before submitting the batch — catches CUDA/OOM/kernel issues the CPU
# smoke cannot (device transfer, pixel-CNN on GPU, real-batch memory footprint).
SMOKE_DEVICE="${SMOKE_DEVICE:-cpu}"

# SMOKE_ONLY restricts which datasets run: "cartpole", "sepsis", "pong", or
# empty (all). Useful to re-test just one dataset without re-running all 12,
# e.g. SMOKE_ONLY=pong after scaling up interactive RAM.
SMOKE_ONLY="${SMOKE_ONLY:-}"
_want() { [ -z "$SMOKE_ONLY" ] || [ "$SMOKE_ONLY" = "$1" ]; }

if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    echo "Skipping smoke runs (SKIP_SMOKE=1)."
else
    echo "Smoke device: $SMOKE_DEVICE${SMOKE_ONLY:+  (only: $SMOKE_ONLY)}"
    ALGOS="discrete_bc discrete_cql discrete_dt discrete_tacr"
    SMOKE_FAILED=0

    for ALGO in $ALGOS; do
        if _want cartpole; then
            for DS in cartpole cartpole_random; do
                echo ""
                echo "--- smoke: $ALGO $DS ---"
                "$PYTHON" "$REPO_DIR/training/train_benchmarks.py" \
                    --algo "$ALGO" --dataset "$DS" --seed 0 --device "$SMOKE_DEVICE" --smoke \
                    && echo "OK: $ALGO $DS" || { echo "FAIL: $ALGO $DS" >&2; SMOKE_FAILED=1; }
            done
        fi

        if _want sepsis; then
            # smoke both modes on terminal (tune=dev/val, final=dev/locked-test),
            # and final on mixed — exercises the 3-way split + both eval paths.
            echo ""
            echo "--- smoke: $ALGO sepsis (tune, terminal) ---"
            "$PYTHON" "$REPO_DIR/training/train_sepsis.py" \
                --algo "$ALGO" --seed 0 --mode tune --device "$SMOKE_DEVICE" --reward_mode terminal --smoke \
                && echo "OK: $ALGO sepsis tune terminal" || { echo "FAIL: $ALGO sepsis tune terminal" >&2; SMOKE_FAILED=1; }

            echo ""
            echo "--- smoke: $ALGO sepsis (final, terminal) ---"
            "$PYTHON" "$REPO_DIR/training/train_sepsis.py" \
                --algo "$ALGO" --seed 0 --mode final --device "$SMOKE_DEVICE" --reward_mode terminal --smoke \
                && echo "OK: $ALGO sepsis final terminal" || { echo "FAIL: $ALGO sepsis final terminal" >&2; SMOKE_FAILED=1; }

            echo ""
            echo "--- smoke: $ALGO sepsis (final, mixed) ---"
            "$PYTHON" "$REPO_DIR/training/train_sepsis.py" \
                --algo "$ALGO" --seed 0 --mode final --device "$SMOKE_DEVICE" --reward_mode mixed --smoke \
                && echo "OK: $ALGO sepsis final mixed" || { echo "FAIL: $ALGO sepsis final mixed" >&2; SMOKE_FAILED=1; }
        fi
    done

    if _want pong; then
        # Pong (minari) is heavy to load; smoke-test once with the transformer actor
        # (the CNN-embedding path most likely to break), not for every algo.
        echo ""
        echo "--- smoke: discrete_tacr pong_minari ---"
        "$PYTHON" "$REPO_DIR/training/train_benchmarks.py" \
            --algo discrete_tacr --dataset pong_minari --seed 0 --device "$SMOKE_DEVICE" --smoke \
            && echo "OK: discrete_tacr pong_minari" || { echo "FAIL: discrete_tacr pong_minari" >&2; SMOKE_FAILED=1; }
    fi

    if [ "$SMOKE_FAILED" = "1" ]; then
        echo ""
        echo "=== SMOKE RUNS FAILED — do NOT submit batch jobs ===" >&2
        exit 1
    fi
fi

echo ""
echo "=== ALL CHECKS PASSED ==="
