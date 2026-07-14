#!/usr/bin/env python3
"""Evaluate a trained DT/TACR policy on sepsis across fixed RTG targets.

Loads a trained algorithm and runs FQE with different fixed return-to-go
values, on the held-out test fold. Tests H1: does TACR's critic make it
less sensitive to the choice of target RTG than plain DecisionTransformer?

Usage:
    python training/eval_rtg_sweep.py \
        --model_path <trained_algo.pt> \
        --algo discrete_tacr \
        --reward_mode mixed \
        --fold 0 \
        --rtg_targets 10 15 20 25

Results saved to eval_logs/{algo}_{reward_mode}_fold{fold}_rtg_sweep.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import d3rlpy
import d3rlpy.preprocessing
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_fold


def evaluate_policy_with_rtg(
    algo,
    train_dataset,
    test_dataset,
    algo_name: str,
    rtg_target,
    device: str = "cpu",
) -> float:
    """Fit FQE on train fold with a fixed RTG target, evaluate on test fold.

    Args:
        algo: Trained algorithm
        train_dataset: Replay buffer to fit FQE's critic on
        test_dataset: Held-out replay buffer to report Q-estimates on
        algo_name: One of {discrete_bc, discrete_cql, discrete_dt, discrete_tacr}
        rtg_target: Fixed return-to-go target (None = use logged trajectory RTG)
        device: CPU or CUDA device

    Returns:
        Mean estimated Q-value from FQE on the test fold
    """
    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")
    if not is_transformer:
        raise ValueError("RTG sweep only applies to RTG-conditioned algos (discrete_dt, discrete_tacr)")

    fqe_config = FQEConfig(batch_size=64, n_critics=1, target_return=rtg_target)
    fqe = DiscreteFQETrajectory(algo, fqe_config, device=device)

    print(f"  Running FQE (RTG={rtg_target})...")
    fqe.fit(
        train_dataset,
        n_steps=50_000,
        n_steps_per_epoch=1_000,
        experiment_name=f"fqe_rtg_{rtg_target}",
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
        batch_type="trajectory",
    )

    # Estimate Q-values on the held-out TEST fold
    obs_list, action_list = [], []
    for episode in test_dataset.episodes:
        obs = episode.observations
        acts = episode.actions
        if isinstance(obs, np.ndarray):
            obs_list.append(obs)
            action_list.append(acts)

    if not obs_list:
        return float("nan")

    all_obs = np.concatenate(obs_list, axis=0)
    all_acts = np.concatenate(action_list, axis=0).squeeze(-1)
    q_values = fqe.predict_value(all_obs, all_acts)
    mean_q = float(np.mean(q_values))

    print(f"    FQE mean Q (test fold): {mean_q:.4f}")
    return mean_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--algo", required=True,
                        choices=["discrete_dt", "discrete_tacr"],
                        help="RTG sweep only applies to RTG-conditioned sequence models")
    parser.add_argument("--reward_mode", default="mixed",
                        choices=["terminal", "dense", "mixed"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--rtg_targets", nargs="+", type=float, default=[10, 15, 20, 25],
                        help="RTG targets to sweep")
    parser.add_argument("--data_dir", default=None,
                        help="Path to MIMIC-IV sepsis CSV (default: $SEPSIS_DATA_DIR)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = args.data_dir or os.environ.get("SEPSIS_DATA_DIR")
    if not data_dir:
        print("ERROR: set --data_dir or SEPSIS_DATA_DIR", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    algo = d3rlpy.load_learnable(args.model_path, device=args.device)

    print(f"Loading sepsis dataset ({args.reward_mode} reward, fold {args.fold}/{args.n_folds})...")
    train_dataset, test_dataset = get_sepsis_fold(
        data_dir=data_dir,
        fold=args.fold,
        n_splits=args.n_folds,
        reward_mode=args.reward_mode,
    )
    print(f"  train: episodes={len(train_dataset.episodes)} transitions={train_dataset.transition_count}")
    print(f"  test:  episodes={len(test_dataset.episodes)} transitions={test_dataset.transition_count}")

    # Run FQE for each RTG target
    results = []
    for rtg in args.rtg_targets:
        print(f"\nRTG target: {rtg}")
        mean_q = evaluate_policy_with_rtg(
            algo, train_dataset, test_dataset, args.algo, rtg, device=args.device
        )
        results.append({"rtg_target": rtg, "fqe_mean_q": mean_q})

    # Save results
    os.makedirs("eval_logs", exist_ok=True)
    result_df = pd.DataFrame(results)
    result_file = f"eval_logs/{args.algo}_{args.reward_mode}_fold{args.fold}_rtg_sweep.csv"
    result_df.to_csv(result_file, index=False)
    print(f"\nResults saved to {result_file}")
    print(result_df.to_string(index=False))
    print(f"\nRTG sensitivity (std across sweep): {result_df['fqe_mean_q'].std():.4f}")


if __name__ == "__main__":
    main()
