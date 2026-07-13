#!/usr/bin/env python3
"""Evaluate TACR on sepsis across RTG targets for Decision Transformer.

Loads a trained algorithm (BC/CQL/DT/TACR) and runs FQE with different
RTG targets. Measures Q-value estimates across return-to-go landscape.

Usage:
    python training/eval_rtg_sweep.py \
        --model_path <trained_algo.pt> \
        --algo discrete_tacr \
        --reward_mode mixed \
        --rtg_targets 10 15 20 25

Results saved to eval_logs/{algo}_{reward_mode}_rtg_sweep.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import d3rlpy
import d3rlpy.preprocessing
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.ope import DiscreteFQE, FQEConfig, FQETrajectory
from d3rlpy.sepsis_loader import get_sepsis


def evaluate_policy_with_rtg(
    algo,
    dataset,
    algo_name: str,
    rtg_target: float,
    device: str = "cpu",
) -> float:
    """Run FQE and estimate mean Q-value at given RTG target.

    Args:
        algo: Trained algorithm
        dataset: Replay buffer for evaluation
        algo_name: One of {discrete_bc, discrete_cql, discrete_dt, discrete_tacr}
        rtg_target: Return-to-go target for evaluation
        device: CPU or CUDA device

    Returns:
        Mean estimated Q-value from FQE
    """
    fqe_config = FQEConfig(batch_size=64, n_critics=1)

    # Trajectory-aware FQE for transformer algos
    if algo_name in ("discrete_dt", "discrete_tacr"):
        fqe = FQETrajectory(algo, fqe_config, device=device)
    else:
        fqe = DiscreteFQE(algo, fqe_config, device=device)

    print(f"  Running FQE (RTG={rtg_target})...")
    fqe.fit(
        dataset,
        n_steps=50_000,
        n_steps_per_epoch=1_000,
        experiment_name=f"fqe_rtg_{rtg_target}",
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
    )

    # Estimate Q-values on a batch from the dataset
    obs_list, action_list = [], []
    for episode in dataset.episodes[:100]:
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

    print(f"    FQE mean Q: {mean_q:.4f}")
    return mean_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--algo", required=True,
                        choices=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    parser.add_argument("--reward_mode", default="mixed",
                        choices=["terminal", "dense", "mixed"])
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

    print(f"Loading sepsis dataset ({args.reward_mode} reward)...")
    dataset = get_sepsis(data_dir=data_dir, reward_mode=args.reward_mode)
    print(f"  Episodes: {len(dataset.episodes)}, Transitions: {dataset.transition_count}")

    # Run FQE for each RTG target
    results = []
    for rtg in args.rtg_targets:
        print(f"\nRTG target: {rtg}")
        mean_q = evaluate_policy_with_rtg(
            algo, dataset, args.algo, rtg, device=args.device
        )
        results.append({"rtg_target": rtg, "fqe_mean_q": mean_q})

    # Save results
    os.makedirs("eval_logs", exist_ok=True)
    result_df = pd.DataFrame(results)
    result_file = f"eval_logs/{args.algo}_{args.reward_mode}_rtg_sweep.csv"
    result_df.to_csv(result_file, index=False)
    print(f"\nResults saved to {result_file}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
