import argparse
import os
import pickle

import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset.components import Episode
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
from d3rlpy.logging import UnifiedFileAdapterFactory
import torch
import time





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. hopper-medium-expert-v2")
    parser.add_argument("--prefix_experiment_name", type=str, default="", help="Prefix of experiment_name, e.g. cpu_array")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--compile", action="store_true", help="Compile model graph")
    parser.add_argument("--n_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of steps per epoch")
    parser.add_argument("--save_interval", type=int, default=1, help="Model save interval")
    parser.add_argument("--eval_gaps", type=int, default=1, help="Evaluation interval in epochs")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of evaluation trials")
    parser.add_argument("--patience", type=int, default=15, help="Number of evaluation trials")

    return parser.parse_args()

def resolve_env_name(dataset):
    if "halfcheetah" in dataset:
        return "HalfCheetah-v4", 6000
    elif "hopper" in dataset:
        return "Hopper-v4", 3800
    elif "walker" in dataset:
        return "Walker2d-v4", 5000
    else:
        raise ValueError("Unsupported dataset name.")

def convert_raw_episode(raw_ep):
    observations = np.array(raw_ep["observations"])
    actions = np.array(raw_ep["actions"])
    rewards = np.array(raw_ep["rewards"])
    if rewards.ndim == 1:
        rewards = rewards.reshape(-1, 1)
    terminals = raw_ep["terminals"]
    terminated = bool(terminals[-1]) if isinstance(terminals, (list, np.ndarray)) else bool(terminals)
    return Episode(observations=observations, actions=actions, rewards=rewards, terminated=terminated)

def load_and_convert_episodes(pkl_path):
    with open(pkl_path, "rb") as f:
        raw_episodes = pickle.load(f)
    return [convert_raw_episode(ep) for ep in raw_episodes]

def main():
    start_time = time.time()
    args = parse_args()

    env_name, target_return = resolve_env_name(args.dataset)
    pkl_path = f"/gpfs/data/fs72297/jklotz/programming/cloned_repos/master_thesis/reproducing_decision_transformer/gymnasium/data/{args.dataset}.pkl"
    #pkl_path = f"../../../../master_thesis/reproducing_decision_transformer/gymnasium/data/{args.dataset}.pkl"
    print(f"Using environment: {env_name}")

    episodes = load_and_convert_episodes(pkl_path)
    print(f"Loaded {len(episodes)} episodes.")

    buffer_impl = FIFOBuffer(limit=10_000_000)
    replay_buffer = ReplayBuffer(buffer=buffer_impl, episodes=episodes)

    env = gym.make(env_name)
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    dt = d3rlpy.algos.DecisionTransformerConfig(
        batch_size=64,
        learning_rate=1e-4,
        optim_factory=d3rlpy.optimizers.AdamWFactory(
            weight_decay=1e-4,
            clip_grad_norm=0.25,
            lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                warmup_steps=10000
            ),
        ),
        encoder_factory=d3rlpy.models.VectorEncoderFactory(
            [128], exclude_last_activation=True
        ),
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MultiplyRewardScaler(0.001),
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        context_size=20,
        num_heads=1,
        num_layers=3,
        max_timestep=1000,
        compile_graph=args.compile,
    ).create(device = "cuda" if torch.cuda.is_available() else "cpu")

    dt.fit(
        replay_buffer,
        
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_epochs,
        save_interval=args.save_interval,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=f"{args.prefix_experiment_name}DT_{args.dataset}_{args.seed}",
        n_trials=args.n_trials,
        eval_gaps=args.eval_gaps,
        logger_adapter=UnifiedFileAdapterFactory(),
        patience=args.patience,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Print or log nicely
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()