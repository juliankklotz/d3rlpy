#!/usr/bin/env python3
"""Hyperparameter tuning for sepsis: grid search on the DEV set, select by VAL FQE.

Leakage-free: uses get_sepsis_splits, which locks a 20% test set (never touched
here) and carves a validation holdout from the 80% development set. Each grid
config is trained on dev-train and scored by the correct per-trajectory FQE
policy value (fqe_policy_value.py) on the validation set. The best config per
algorithm (per reward mode) is written to tuned_configs.json, which
train_sepsis.py --mode final then consumes.

This is the real tuner (vs tune_sepsis_safeguard.py, which was only a stability
sanity check). It uses reduced training/FQE steps to rank configs — the FINAL
run (train_sepsis.py --mode final) retrains the chosen config at full length.

Usage:
    SEPSIS_DATA_DIR=/path python training/tune_sepsis.py --device cuda:0 \
        --reward_mode terminal --n_steps 50000 --fqe_n_steps 50000
    # then again with --reward_mode mixed
    # merges both into tuned_configs.json

Grids (reduced, from safeguard findings — configs flip FQE sign, so tuning
matters). BC lr, CQL alpha, DT layers, TACR alpha x layers.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d3rlpy
import d3rlpy.preprocessing
import d3rlpy.models
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_splits

from fqe_policy_value import fqe_policy_value_per_trajectory

# import build_algo + constants from train_sepsis to guarantee identical config
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_sepsis", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_sepsis.py")
)
train_sepsis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_sepsis)
build_algo = train_sepsis.build_algo
CONTEXT_SIZE = train_sepsis.CONTEXT_SIZE
EVAL_RTG_BY_REWARD_MODE = train_sepsis.EVAL_RTG_BY_REWARD_MODE

OUT_JSON = "tuned_configs.json"

# Reduced grids per algo — keys must match the *Config kwargs merged in build_algo.
GRIDS = {
    "discrete_bc": [{"learning_rate": lr} for lr in (1e-3, 3e-4)],
    "discrete_cql": [{"alpha": a} for a in (1.0, 4.0)],
    "discrete_dt": [
        {"num_layers": nl} for nl in (3, 6)
    ],
    "discrete_tacr": [
        {"alpha": a, "num_layers": nl}
        for a, nl in itertools.product((0.5, 1.0, 2.5), (3, 6))
    ],
}


def evaluate_config(algo_name, hp, dev_train, val, reward_mode, device,
                    n_steps, fqe_n_steps):
    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")
    algo = build_algo(algo_name, device, hp_override=hp)

    losses = algo.fit(
        dev_train, n_steps=n_steps, n_steps_per_epoch=max(1, n_steps // 5),
        logger_adapter=NoopAdapterFactory(), show_progress=False,
    )
    last = losses[-1][1] if losses else {}
    stable = all(np.isfinite(v) for v in last.values() if isinstance(v, (int, float)))

    target_rtg = EVAL_RTG_BY_REWARD_MODE[reward_mode] if is_transformer else None
    fqe_cfg = FQEConfig(batch_size=64, n_critics=1, target_return=target_rtg)
    if is_transformer:
        fqe = DiscreteFQETrajectory(algo, fqe_cfg, device=device)
        fqe.fit(dev_train, n_steps=fqe_n_steps, n_steps_per_epoch=max(1, fqe_n_steps // 5),
                logger_adapter=NoopAdapterFactory(), show_progress=False, batch_type="trajectory")
    else:
        fqe = DiscreteFQE(algo, fqe_cfg, device=device)
        fqe.fit(dev_train, n_steps=fqe_n_steps, n_steps_per_epoch=max(1, fqe_n_steps // 5),
                logger_adapter=NoopAdapterFactory(), show_progress=False)

    v = fqe_policy_value_per_trajectory(fqe, algo, val, algo_name, target_rtg)
    return (float(np.mean(v)) if len(v) else float("nan")), stable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--reward_mode", default="terminal", choices=["terminal", "dense", "mixed"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--n_steps", type=int, default=50_000)
    p.add_argument("--fqe_n_steps", type=int, default=50_000)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--algos", nargs="+",
                   default=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    p.add_argument("--out", default=OUT_JSON)
    args = p.parse_args()

    data_dir = args.data_dir or os.environ.get("SEPSIS_DATA_DIR")
    if not data_dir:
        print("ERROR: set --data_dir or SEPSIS_DATA_DIR", file=sys.stderr)
        sys.exit(1)

    d3rlpy.seed(0)
    print(f"[TUNE {args.reward_mode}] loading dev-train / val (locked test untouched) ...")
    dev_train, val, _locked_test = get_sepsis_splits(
        data_dir=data_dir, test_frac=args.test_frac, holdout=True,
        val_frac=args.val_frac, seed=args.split_seed, reward_mode=args.reward_mode,
    )
    print(f"  dev-train: {len(dev_train.episodes)} eps   val: {len(val.episodes)} eps")

    # merge into existing JSON if present (so terminal + mixed accumulate)
    tuned = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            tuned = json.load(f)

    results = []
    for algo_name in args.algos:
        print(f"\n=== {algo_name} ({args.reward_mode}) ===")
        best = (None, -np.inf)
        for hp in GRIDS[algo_name]:
            try:
                v_hat, stable = evaluate_config(
                    algo_name, hp, dev_train, val, args.reward_mode,
                    args.device, args.n_steps, args.fqe_n_steps,
                )
            except Exception as e:
                print(f"  {hp} -> ERROR: {e}")
                results.append({"algo": algo_name, "reward": args.reward_mode,
                                "hp": str(hp), "val_fqe": float("nan"), "stable": False})
                continue
            flag = "" if stable else " [UNSTABLE]"
            print(f"  {hp} -> val FQE {v_hat:.4f}{flag}")
            results.append({"algo": algo_name, "reward": args.reward_mode,
                            "hp": str(hp), "val_fqe": v_hat, "stable": stable})
            if stable and v_hat > best[1]:
                best = (hp, v_hat)
        if best[0] is not None:
            print(f"  BEST: {best[0]}  (val FQE {best[1]:.4f})")
            tuned.setdefault(algo_name, {})[args.reward_mode] = best[0]
        else:
            print(f"  !! no stable config for {algo_name} — leaving unset")

    with open(args.out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"\nTuned configs written to {args.out}")
    print(json.dumps(tuned, indent=2))

    import pandas as pd
    os.makedirs("tune_logs", exist_ok=True)
    csv = f"tune_logs/tune_{args.reward_mode}.csv"
    pd.DataFrame(results).to_csv(csv, index=False)
    print(f"Full grid results: {csv}")


if __name__ == "__main__":
    main()
