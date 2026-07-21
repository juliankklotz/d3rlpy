#!/usr/bin/env python3
"""Cheap hyperparameter safeguard: does ANY config train sanely, on fold 0 only?

Purpose: before committing to the full 5-fold x 3-seed x 2-reward batch (120
sepsis jobs), verify on a SINGLE fold that each algorithm's hyperparameters are
in a sensible range — i.e. training is stable (losses finite, not diverging) and
the FQE policy value is non-degenerate. Catches "all my HPs are bad, redo
everything" early, for ~a handful of short jobs instead of 120 full ones.

This is NOT the final tuning. It is a small grid per algorithm, run at reduced
n_steps, selecting the best config by the correct FQE policy-value metric
(training/fqe_policy_value.py) on a validation split carved from fold-0's train
set. Fold 0's test set is left untouched (no leakage into any later reported
result if fold 0 is reserved for validation).

Usage:
    SEPSIS_DATA_DIR=/path python training/tune_sepsis_safeguard.py --device cuda:0
    python training/tune_sepsis_safeguard.py --device cuda:0 --reward_mode terminal --n_steps 20000

Output: prints, per algo, each config's val FQE value + whether it trained
stably, and the best config. Writes a CSV summary.
"""
import argparse
import itertools
import os
import sys

import numpy as np

import d3rlpy
import d3rlpy.preprocessing
import d3rlpy.models
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_fold

sys.path.insert(0, os.path.dirname(__file__))
from fqe_policy_value import fqe_policy_value_per_trajectory

N_ACTIONS = 25
OBS_DIM = 47
CONTEXT_SIZE = 20  # must match FQE trajectory sampler's hardcoded length=20

EVAL_RTG = {"terminal": 1.0, "mixed": 15.0, "dense": None}

# Small grids — a few points per algo, NOT the full thesis grid. Just enough to
# see whether the sane region is being hit.
GRIDS = {
    "discrete_bc": [
        {"learning_rate": 1e-3},
        {"learning_rate": 3e-4},
    ],
    "discrete_cql": [
        {"alpha": 1.0},
        {"alpha": 4.0},
    ],
    "discrete_dt": [
        {"context_size": CONTEXT_SIZE, "num_heads": 4, "num_layers": 3},
        {"context_size": CONTEXT_SIZE, "num_heads": 4, "num_layers": 6},
    ],
    "discrete_tacr": [
        {"context_size": CONTEXT_SIZE, "num_heads": 4, "num_layers": 3, "alpha": 0.5},
        {"context_size": CONTEXT_SIZE, "num_heads": 4, "num_layers": 3, "alpha": 2.5},
    ],
}


def build_algo(algo_name, hp, device):
    obs_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    if algo_name == "discrete_bc":
        return d3rlpy.algos.DiscreteBCConfig(
            batch_size=64, observation_scaler=obs_scaler, **hp
        ).create(device=device)
    if algo_name == "discrete_cql":
        return d3rlpy.algos.DiscreteCQLConfig(
            batch_size=64, observation_scaler=obs_scaler, **hp
        ).create(device=device)
    if algo_name == "discrete_dt":
        return d3rlpy.algos.DiscreteDecisionTransformerConfig(
            batch_size=64, observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            max_timestep=100, **hp,
        ).create(device=device)
    if algo_name == "discrete_tacr":
        return d3rlpy.algos.DiscreteTACRConfig(
            batch_size=64, observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            actor_learning_rate=1e-4, max_timestep=100,
            actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                [256, 256], exclude_last_activation=True
            ),
            actor_optim_factory=d3rlpy.optimizers.AdamWFactory(
                weight_decay=1e-4, clip_grad_norm=0.25,
                lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                    warmup_steps=1_000
                ),
            ),
            position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
            compile_graph=False, **hp,
        ).create(device=device)
    raise ValueError(algo_name)


def split_train_val(buffer, val_frac=0.2, seed=0):
    """Split a ReplayBuffer's episodes into train/val by episode."""
    from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
    rng = np.random.default_rng(seed)
    eps = list(buffer.episodes)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac))
    val_eps, train_eps = eps[:n_val], eps[n_val:]

    def mk(es):
        return ReplayBuffer(
            buffer=FIFOBuffer(limit=sum(len(e) for e in es)),
            episodes=es,
            action_space=buffer.dataset_info.action_space,
            action_size=buffer.dataset_info.action_size,
        )
    return mk(train_eps), mk(val_eps)


def evaluate_config(algo_name, hp, train_ds, val_ds, reward_mode, device,
                    n_steps, fqe_n_steps):
    """Train one config, FQE-evaluate on val split, return (V_hat, stable)."""
    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")
    algo = build_algo(algo_name, hp, device)

    losses = algo.fit(
        train_ds, n_steps=n_steps, n_steps_per_epoch=max(1, n_steps // 5),
        logger_adapter=NoopAdapterFactory(), show_progress=False,
    )
    # stability: last-epoch losses finite
    last = losses[-1][1] if losses else {}
    stable = all(np.isfinite(v) for v in last.values() if isinstance(v, (int, float)))

    target_rtg = EVAL_RTG[reward_mode] if is_transformer else None
    fqe_cfg = FQEConfig(batch_size=64, n_critics=1, target_return=target_rtg)
    if is_transformer:
        fqe = DiscreteFQETrajectory(algo, fqe_cfg, device=device)
        fqe.fit(train_ds, n_steps=fqe_n_steps, n_steps_per_epoch=max(1, fqe_n_steps // 5),
                logger_adapter=NoopAdapterFactory(), show_progress=False,
                batch_type="trajectory")
    else:
        fqe = DiscreteFQE(algo, fqe_cfg, device=device)
        fqe.fit(train_ds, n_steps=fqe_n_steps, n_steps_per_epoch=max(1, fqe_n_steps // 5),
                logger_adapter=NoopAdapterFactory(), show_progress=False)

    v = fqe_policy_value_per_trajectory(fqe, algo, val_ds, algo_name, target_rtg)
    v_hat = float(np.mean(v)) if len(v) else float("nan")
    return v_hat, stable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--reward_mode", default="terminal", choices=["terminal", "dense", "mixed"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--n_steps", type=int, default=20_000, help="reduced train steps for the safeguard")
    p.add_argument("--fqe_n_steps", type=int, default=10_000)
    p.add_argument("--algos", nargs="+",
                   default=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    args = p.parse_args()

    data_dir = args.data_dir or os.environ.get("SEPSIS_DATA_DIR")
    if not data_dir:
        print("ERROR: set --data_dir or SEPSIS_DATA_DIR", file=sys.stderr)
        sys.exit(1)

    d3rlpy.seed(0)
    print(f"Loading fold 0 ({args.reward_mode}) ...")
    fold0_train, _fold0_test = get_sepsis_fold(
        data_dir=data_dir, fold=0, n_splits=5, reward_mode=args.reward_mode
    )
    # carve validation from fold-0 train; fold-0 test is untouched
    train_ds, val_ds = split_train_val(fold0_train, val_frac=0.2, seed=0)
    print(f"  tune-train: {len(train_ds.episodes)} eps   val: {len(val_ds.episodes)} eps")

    rows = []
    for algo_name in args.algos:
        print(f"\n=== {algo_name} ===")
        best = (None, -np.inf, False)
        for hp in GRIDS[algo_name]:
            try:
                v_hat, stable = evaluate_config(
                    algo_name, hp, train_ds, val_ds, args.reward_mode,
                    args.device, args.n_steps, args.fqe_n_steps,
                )
            except Exception as e:
                print(f"  {hp}  -> ERROR: {e}")
                rows.append({"algo": algo_name, "hp": str(hp),
                             "val_fqe": float("nan"), "stable": False, "error": str(e)})
                continue
            flag = "stable" if stable else "UNSTABLE(non-finite loss)"
            print(f"  {hp}  ->  val FQE V_hat={v_hat:.4f}  [{flag}]")
            rows.append({"algo": algo_name, "hp": str(hp),
                         "val_fqe": v_hat, "stable": stable, "error": ""})
            if stable and v_hat > best[1]:
                best = (hp, v_hat, True)
        if best[0] is not None:
            print(f"  BEST: {best[0]}  (val FQE {best[1]:.4f})")
        else:
            print(f"  !! no stable config for {algo_name} — HPs likely bad")

    import pandas as pd
    os.makedirs("tune_logs", exist_ok=True)
    out = f"tune_logs/safeguard_{args.reward_mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSummary written to {out}")

    # verdict
    per_algo_ok = {}
    for algo_name in args.algos:
        algo_rows = [r for r in rows if r["algo"] == algo_name]
        per_algo_ok[algo_name] = any(r["stable"] for r in algo_rows)
    print("\n=== VERDICT ===")
    for a, ok in per_algo_ok.items():
        print(f"  {a}: {'OK (>=1 stable config)' if ok else 'PROBLEM (no stable config)'}")
    if all(per_algo_ok.values()):
        print("All algos have a sane config on fold 0 — safe to design the full run.")
    else:
        print("Some algos have NO stable config — fix HPs before the full batch.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
