#!/usr/bin/env python3
"""Plot FQE critic loss convergence and suggest an FQE_N_STEPS cutoff.

Reads d3rlpy_logs/<experiment_name>/metrics.csv (written by
UnifiedFileAdapterFactory during fqe.fit()) and plots loss vs step.
Plateau is detected via relative change of a trailing moving average.

Usage:
    python training/plot_fqe_convergence.py --experiment fqe_discrete_tacr_sepsis_fold0_seed0
    python training/plot_fqe_convergence.py --experiment fqe_discrete_tacr_sepsis_fold0_seed0 \
        --window 20 --rel_tol 0.01 --out fqe_convergence.png

    # compare multiple runs on one plot:
    python training/plot_fqe_convergence.py --experiment fqe_discrete_bc_sepsis_fold0_seed0 \
        fqe_discrete_cql_sepsis_fold0_seed0 fqe_discrete_tacr_sepsis_fold0_seed0
"""
import argparse
import os

import numpy as np
import pandas as pd


def detect_plateau(steps: np.ndarray, loss: np.ndarray, window: int, rel_tol: float) -> int | None:
    """Return the step at which the trailing moving average stops improving
    by more than rel_tol (relative change), or None if never plateaus.
    """
    if len(loss) < 2 * window:
        return None

    smoothed = pd.Series(loss).rolling(window, min_periods=window).mean().to_numpy()

    for i in range(2 * window, len(smoothed)):
        prev = smoothed[i - window]
        curr = smoothed[i]
        if prev == 0:
            continue
        rel_change = abs(curr - prev) / abs(prev)
        if rel_change < rel_tol:
            return int(steps[i])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", required=True,
                        help="Experiment name(s) under d3rlpy_logs/ (fqe_ prefix included)")
    parser.add_argument("--logdir", default="d3rlpy_logs",
                        help="Root log directory (default: d3rlpy_logs)")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling window size (in epochs) for plateau smoothing")
    parser.add_argument("--rel_tol", type=float, default=0.01,
                        help="Relative change threshold to call it plateaued (default 1%%)")
    parser.add_argument("--out", default=None,
                        help="Save plot to file instead of showing interactively")
    args = parser.parse_args()

    # Plotting is optional — plateau DETECTION (the number you need) works without
    # matplotlib. Only build a figure if matplotlib is importable.
    plt = None
    ax = None
    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _fig, ax = plt.subplots(figsize=(9, 5))
    except ModuleNotFoundError:
        print("(matplotlib not installed — printing plateau detection only, no plot)")

    for exp in args.experiment:
        csv_path = os.path.join(args.logdir, exp, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        if "loss" not in df.columns:
            print(f"WARNING: no 'loss' column in {csv_path}, skipping")
            continue

        steps = df["step"].to_numpy()
        loss = df["loss"].to_numpy()

        if ax is not None:
            ax.plot(steps, loss, label=exp, alpha=0.4, linewidth=1)
            smoothed = pd.Series(loss).rolling(args.window, min_periods=1).mean()
            ax.plot(steps, smoothed, label=f"{exp} (smoothed)", linewidth=2)

        plateau_step = detect_plateau(steps, loss, args.window, args.rel_tol)
        if plateau_step is not None:
            if ax is not None:
                ax.axvline(plateau_step, linestyle="--", alpha=0.5)
            print(f"{exp}: plateaus at step {plateau_step} "
                  f"(rel_tol={args.rel_tol}, window={args.window}); "
                  f"final loss={loss[-1]:.4f}")
            print(f"  -> suggest FQE_N_STEPS = {plateau_step} "
                  f"(or {int(plateau_step * 1.2)} with 20% safety margin)")
        else:
            print(f"{exp}: no plateau detected within logged range "
                  f"(max step={steps.max()}, final loss={loss[-1]:.4f}). "
                  f"Loss still moving — FQE_N_STEPS may need to be higher.")

    if ax is not None:
        ax.set_xlabel("FQE training step")
        ax.set_ylabel("Critic loss (MSE Bellman error)")
        ax.set_title("FQE convergence")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _fig.tight_layout()
        if args.out:
            _fig.savefig(args.out, dpi=150)
            print(f"Saved plot to {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
