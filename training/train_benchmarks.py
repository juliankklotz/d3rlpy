#!/usr/bin/env python3
"""Train one discrete offline RL algo on CartPole or Pong with online eval.

Usage:
    python training/train_benchmarks.py --algo discrete_bc --dataset cartpole --seed 0
    python training/train_benchmarks.py --algo discrete_tacr --dataset pong --seed 1 --device cuda:0
"""
import argparse
import os
import sys

import d3rlpy
import d3rlpy.preprocessing
import d3rlpy.models
from d3rlpy.logging import UnifiedFileAdapterFactory


# ── hyperparameters ────────────────────────────────────────────────────────────
# One set per (algo, dataset). Derived from exp08 for TACR; literature for others.
HPARAMS = {
    "cartpole": {
        "target_return": 200,
        "n_steps": 100_000,
        "n_steps_per_epoch": 1_000,
        "discrete_bc": dict(batch_size=64, learning_rate=1e-3),
        "discrete_cql": dict(batch_size=64),
        "discrete_dt": dict(
            batch_size=64, context_size=20, num_heads=1, num_layers=3,
            max_timestep=1_000,
        ),
        "discrete_tacr": dict(
            batch_size=64, context_size=20, num_heads=1, num_layers=3,
            actor_learning_rate=1e-4, max_timestep=1_000,
            position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
            compile_graph=False,
        ),
    },
    "pong": {
        # dataset: get_atari("pong-expert-v0", num_stack=4) → obs [4, 84, 84]
        "target_return": 20,
        "n_steps": 500_000,
        "n_steps_per_epoch": 5_000,
        "discrete_bc": dict(batch_size=64, learning_rate=1e-3),
        "discrete_cql": dict(batch_size=64),
        "discrete_dt": dict(
            batch_size=64, context_size=30, num_heads=4, num_layers=6,
            max_timestep=27_000,
        ),
        "discrete_tacr": dict(
            batch_size=64, context_size=30, num_heads=4, num_layers=6,
            actor_learning_rate=1e-4, max_timestep=27_000,
            position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
            compile_graph=False,
            # alpha=0.5 is fine; alpha normalizer clamped in tacr_impl
        ),
    },
}


PONG_DATASETS = {"pong", "pong_minari"}


def _actor_encoder(dataset_name: str) -> object:
    """Pixel encoder for Pong variants, vector encoder for CartPole."""
    if dataset_name in PONG_DATASETS:
        return d3rlpy.models.PixelEncoderFactory(
            filters=[[32, 8, 4], [64, 4, 2], [64, 3, 1]],
            feature_size=512,
            exclude_last_activation=True,
        )
    return d3rlpy.models.VectorEncoderFactory([128], exclude_last_activation=True)


def _hparam_key(dataset_name: str) -> str:
    # pong_minari uses the same HPs as pong
    return "pong" if dataset_name == "pong_minari" else dataset_name


def _warmup_steps(dataset_name: str) -> int:
    return HPARAMS[_hparam_key(dataset_name)]["n_steps"] // 100


def build_algo(algo_name: str, dataset_name: str, device: str) -> d3rlpy.base.LearnableBase:
    hp = HPARAMS[_hparam_key(dataset_name)][algo_name]
    obs_scaler: object
    if dataset_name in PONG_DATASETS:
        obs_scaler = d3rlpy.preprocessing.PixelObservationScaler()
    else:
        obs_scaler = d3rlpy.preprocessing.StandardObservationScaler()

    if algo_name == "discrete_bc":
        return d3rlpy.algos.DiscreteBCConfig(
            observation_scaler=obs_scaler,
            **hp,
        ).create(device=device)

    if algo_name == "discrete_cql":
        return d3rlpy.algos.DiscreteCQLConfig(
            observation_scaler=obs_scaler,
            **hp,
        ).create(device=device)

    if algo_name == "discrete_dt":
        return d3rlpy.algos.DiscreteDecisionTransformerConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            encoder_factory=_actor_encoder(dataset_name),
            **hp,
        ).create(device=device)

    if algo_name == "discrete_tacr":
        return d3rlpy.algos.DiscreteTACRConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            actor_encoder_factory=_actor_encoder(dataset_name),
            actor_optim_factory=d3rlpy.optimizers.AdamWFactory(
                weight_decay=1e-4,
                clip_grad_norm=0.25,
                lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                    warmup_steps=_warmup_steps(dataset_name)
                ),
            ),
            **hp,
        ).create(device=device)

    raise ValueError(f"Unknown algo: {algo_name}")


def load_dataset(
    dataset_name: str,
) -> tuple[d3rlpy.dataset.ReplayBuffer, object]:
    if dataset_name == "cartpole":
        return d3rlpy.datasets.get_cartpole()
    if dataset_name == "pong":
        # Standard offline Atari: [4, 84, 84] grayscale stacked frames.
        # Requires: pip install "gym[atari,accept-rom-license]" autorom && AutoROM --accept-license
        #           pip install git+https://github.com/takuseno/d4rl-atari
        try:
            return d3rlpy.datasets.get_atari("pong-expert-v0", num_stack=4)
        except ImportError:
            raise ImportError(
                "d4rl_atari not installed. Run:\n"
                "  pip install 'gym[atari,accept-rom-license]' autorom\n"
                "  AutoROM --accept-license\n"
                "  pip install git+https://github.com/takuseno/d4rl-atari\n"
                "Or set --dataset pong_minari to use raw minari frames (suboptimal)."
            )
    if dataset_name == "pong_minari":
        # Fallback: minari raw [3, 210, 160] frames with FrameStack(4) → [12, 210, 160].
        # Larger memory footprint; not standard benchmark format.
        from d3rlpy.dataset import FrameStackTransitionPicker, FrameStackTrajectorySlicer
        return d3rlpy.datasets.get_minari(
            "atari/pong/expert-v0",
            transition_picker=FrameStackTransitionPicker(n_frames=4),
            trajectory_slicer=FrameStackTrajectorySlicer(n_frames=4),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}. Use cartpole, pong, or pong_minari.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        choices=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    parser.add_argument("--dataset", required=True,
                        choices=["cartpole", "pong", "pong_minari"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Override default n_steps for this dataset")
    parser.add_argument("--logdir", default=None,
                        help="Experiment name (default: algo_dataset_seedN)")
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = load_dataset(args.dataset)
    d3rlpy.envs.seed_env(env, args.seed)

    algo = build_algo(args.algo, args.dataset, args.device)

    hp = HPARAMS[_hparam_key(args.dataset)]
    n_steps = args.n_steps or hp["n_steps"]
    experiment_name = args.logdir or f"{args.algo}_{args.dataset}_seed{args.seed}"

    print(f"algo={args.algo}  dataset={args.dataset}  seed={args.seed}  n_steps={n_steps}")
    print(f"python={sys.executable}  device={args.device}")

    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=hp["n_steps_per_epoch"],
        eval_env=env,
        eval_target_return=hp["target_return"],
        save_interval=hp["n_steps_per_epoch"] * 10,
        experiment_name=experiment_name,
        logger_adapter=UnifiedFileAdapterFactory(),
        show_progress=False,
    )


if __name__ == "__main__":
    main()
