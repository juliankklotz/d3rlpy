#!/usr/bin/env python3
"""Train one discrete offline RL algo on the Sepsis/MIMIC-IV dataset, then run FQE.

No simulator available — evaluation is offline-only via FQE, on a held-out
test fold (5-fold stratified CV by survival outcome).
BC/CQL use DiscreteFQE; DT/TACR use DiscreteFQETrajectory (sequence-aware).

Usage:
    python training/train_sepsis.py --algo discrete_bc --seed 0 --fold 0
    python training/train_sepsis.py --algo discrete_tacr --seed 1 --fold 0 --device cuda:0
    SEPSIS_DATA_DIR=/data/sepsis python training/train_sepsis.py --algo discrete_cql --seed 2 --fold 0
"""
import argparse
import os
import sys
from pathlib import Path

import d3rlpy
import d3rlpy.preprocessing
import d3rlpy.models
from d3rlpy.logging import UnifiedFileAdapterFactory
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_fold

# Sepsis dataset constants
N_ACTIONS = 25     # 5 fluid levels × 5 vasopressor levels
OBS_DIM = 47       # see sepsis_loader.OBSERVATION_COLUMNS
CONTEXT_SIZE = 20  # median ICU stay length — MUST match hardcoded FQE trajectory
                    # length=20 in QLearningAlgoBase.fitter() (batch_type="trajectory").
                    # Sequence-length ablations (K != 20) cannot be FQE-evaluated as-is.

HPARAMS = {
    "discrete_bc": dict(batch_size=64, learning_rate=1e-3),
    "discrete_cql": dict(batch_size=64),
    "discrete_dt": dict(
        batch_size=64, context_size=CONTEXT_SIZE, num_heads=4, num_layers=6,
        max_timestep=100,
    ),
    "discrete_tacr": dict(
        batch_size=64, context_size=CONTEXT_SIZE, num_heads=4, num_layers=6,
        actor_learning_rate=1e-4, max_timestep=100,
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        compile_graph=False,
    ),
}

N_STEPS = 100_000
N_STEPS_PER_EPOCH = 1_000
FQE_N_STEPS = 50_000
N_FOLDS = 5

# RTG used to query DT/TACR at FQE bootstrap time, per reward_mode.
# terminal reward: RTG is unambiguous (+1 = desired survival).
# mixed reward: no natural RTG scale: use primary target from thesis 5.2.3.
EVAL_RTG_BY_REWARD_MODE = {
    "terminal": 1.0,
    "mixed": 15.0,
    "dense": None,  # dense reward has no principled fixed target; use data-derived RTG
}


def build_algo(algo_name: str, device: str) -> d3rlpy.base.LearnableBase:
    hp = HPARAMS[algo_name]
    obs_scaler = d3rlpy.preprocessing.StandardObservationScaler()

    if algo_name == "discrete_bc":
        return d3rlpy.algos.DiscreteBCConfig(
            observation_scaler=obs_scaler, **hp
        ).create(device=device)

    if algo_name == "discrete_cql":
        return d3rlpy.algos.DiscreteCQLConfig(
            observation_scaler=obs_scaler, **hp
        ).create(device=device)

    if algo_name == "discrete_dt":
        return d3rlpy.algos.DiscreteDecisionTransformerConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            **hp,
        ).create(device=device)

    if algo_name == "discrete_tacr":
        return d3rlpy.algos.DiscreteTACRConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                [256, 256], exclude_last_activation=True
            ),
            actor_optim_factory=d3rlpy.optimizers.AdamWFactory(
                weight_decay=1e-4,
                clip_grad_norm=0.25,
                lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                    warmup_steps=1_000
                ),
            ),
            **hp,
        ).create(device=device)

    raise ValueError(f"Unknown algo: {algo_name}")


def run_fqe(
    algo: d3rlpy.base.LearnableBase,
    train_dataset: d3rlpy.dataset.ReplayBuffer,
    test_dataset: d3rlpy.dataset.ReplayBuffer,
    algo_name: str,
    reward_mode: str,
    device: str,
    experiment_name: str,
) -> float:
    """Fit FQE on the TRAINING fold, evaluate on the held-out TEST fold.

    FQE itself must be fit on data (any offline data works for fitting the
    Q-function), but the reported estimate must reflect generalisation, so
    the Q-values are predicted for transitions drawn from the test fold.
    """
    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")
    target_rtg = EVAL_RTG_BY_REWARD_MODE[reward_mode] if is_transformer else None

    fqe_config = FQEConfig(batch_size=64, n_critics=1, target_return=target_rtg)

    if is_transformer:
        fqe = DiscreteFQETrajectory(algo, fqe_config, device=device)
        fit_kwargs = dict(batch_type="trajectory")
    else:
        fqe = DiscreteFQE(algo, fqe_config, device=device)
        fit_kwargs = dict()

    fqe.fit(
        train_dataset,
        n_steps=FQE_N_STEPS,
        n_steps_per_epoch=1_000,
        experiment_name=f"fqe_{experiment_name}",
        logger_adapter=UnifiedFileAdapterFactory(),
        show_progress=False,
        **fit_kwargs,
    )

    # Estimate Q-value on the held-out TEST fold (generalisation, not memorisation)
    import numpy as np

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
    print(f"FQE mean Q-value (test fold): {mean_q:.4f}  (n={len(all_obs)} transitions)")
    return mean_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        choices=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0,
                        help="Test fold index in [0, n_folds)")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_dir", default=None,
                        help="Path to MIMIC-IV sepsis CSV (default: $SEPSIS_DATA_DIR)")
    parser.add_argument("--reward_mode", default="terminal",
                        choices=["terminal", "dense", "mixed"],
                        help="terminal: ±1 survival/death; dense: -SOFA each step; mixed: 0.5*SOFA-delta + ±15 terminal")
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--skip_fqe", action="store_true",
                        help="Skip FQE eval (faster, use for debugging)")
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    data_dir = args.data_dir or os.environ.get("SEPSIS_DATA_DIR")
    if not data_dir:
        print("ERROR: set --data_dir or SEPSIS_DATA_DIR", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sepsis dataset from {data_dir} (fold {args.fold}/{args.n_folds}) ...")
    train_dataset, test_dataset = get_sepsis_fold(
        data_dir=data_dir,
        fold=args.fold,
        n_splits=args.n_folds,
        reward_mode=args.reward_mode,
    )
    print(f"  train: episodes={len(train_dataset.episodes)} transitions={train_dataset.transition_count}")
    print(f"  test:  episodes={len(test_dataset.episodes)} transitions={test_dataset.transition_count}")

    algo = build_algo(args.algo, args.device)
    experiment_name = f"{args.algo}_sepsis_fold{args.fold}_seed{args.seed}"

    print(f"algo={args.algo}  seed={args.seed}  fold={args.fold}  n_steps={args.n_steps}")
    print(f"python={sys.executable}  device={args.device}")

    algo.fit(
        train_dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        save_interval=N_STEPS_PER_EPOCH * 10,
        experiment_name=experiment_name,
        logger_adapter=UnifiedFileAdapterFactory(),
        show_progress=False,
    )

    model_path = f"{experiment_name}_final.pt"
    algo.save(model_path)
    print(f"Model saved: {model_path}")

    if not args.skip_fqe:
        print("Running FQE (train fold fit, test fold eval) ...")
        run_fqe(algo, train_dataset, test_dataset, args.algo, args.reward_mode, args.device, experiment_name)


if __name__ == "__main__":
    main()
