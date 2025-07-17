import argparse
import os
import pickle

import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.dataset.components import Episode
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, InfiniteBuffer
from d3rlpy.logging import UnifiedFileAdapterFactory

import argparse
from datetime import datetime
from typing import Optional, Union

#import gym
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.algos import CQL, IQL
from d3rlpy.dataset import , ReplayBuffer
from d3rlpy.types import NDArray

def resolve_env_name(dataset):
    if "halfcheetah" in dataset:
        return "HalfCheetah-v4", 6000
    elif "hopper" in dataset:
        return "Hopper-v4", 3600
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v2")
    parser.add_argument("--context_size", type=int, default=20)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument(
        "--q_learning_type",
        type=str,
        default="cql",
        choices=["cql", "iql"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_action_samples", type=int, default=10)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    env_name, target_return = resolve_env_name(args.dataset)
    pkl_path = f"/gpfs/data/fs72297/jklotz/programming/cloned_repos/master_thesis/reproducing_decision_transformer/gymnasium/data/{args.dataset}.pkl"
    #pkl_path = f"../../../../master_thesis/reproducing_decision_transformer/gymnasium/data/{args.dataset}.pkl"
    print(f"Using environment: {env_name}")

    episodes = load_and_convert_episodes(pkl_path)
    print(f"Loaded {len(episodes)} episodes.")

    buffer_impl = FIFOBuffer(limit=10_000_000)
    replay_buffer = ReplayBuffer(buffer=buffer_impl, episodes=episodes)
    dataset = replay_buffer
    env = gym.make(env_name)

    # create postfix of log directories
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_postfix = f"{env.spec.id}_{args.seed}_{timestamp}"

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # first fit Q-learning algorithm to the dataset
    if args.model_file is not None:
        # load model and assert type
        q_algo_loaded = d3rlpy.load_learnable(args.model_file)
        if not isinstance(q_algo_loaded, (CQL, IQL)):
            raise ValueError(
                "The loaded model is not an instance of CQL or IQL."
            )
        # cast to the expected type
        q_algo = q_algo_loaded
    else:
        if args.q_learning_type == "cql":
            q_algo = fit_cql(
                dataset=dataset,
                env=env,
                gpu=args.gpu,
                log_postfix=log_postfix,
            )
        elif args.q_learning_type == "iql":
            q_algo = fit_iql(
                dataset=dataset,
                env=env,
                gpu=args.gpu,
                log_postfix=log_postfix,
            )
        else:
            raise ValueError(f"invalid q_learning_type: {args.q_learning_type}")

    # relabel dataset RTGs with the learned value functions
    print("Relabeling dataset with RTGs...")
    assert isinstance(dataset._buffer, InfiniteBuffer)
    relabel_dataset_rtg(
        buffer=dataset._buffer,
        q_algo=q_algo,
        k=args.context_size,
        num_action_samples=args.num_action_samples,
    )

    # fit decision transformer to the relabeled dataset
    fit_dt(
        dataset=dataset,
        env=env,
        context_size=args.context_size,
        gpu=args.gpu,
        log_postfix=log_postfix,
    )


""" --------------------------------------------------------------------
    Augment dataset
-------------------------------------------------------------------- """


def relabel_dataset_rtg(
    buffer: InfiniteBuffer,
    q_algo: Union[CQL, IQL],
    k: int,
    num_action_samples: int,
) -> None:
    """
    Relabel RTG (reward-to-go) to the given dataset using the given Q-function.

    Args:
        buffer (InfiniteBuffer): Buffer holding trajectory dataset.
        q_algo (Union[CQL, IQL]): Trained Q-learning algoirthm.
        k (int): Context length for DT.
        num_action_samples (int): The number of action samples for
            V function estimation. Defaults to 10.
    """
    prev_idx = -1
    for n in range(buffer.transition_count):
        episode, idx = buffer._transitions[-n - 1]  # get transitions backwards
        if idx > prev_idx:
            # get values for all observations in the episode
            values = []
            for _ in range(num_action_samples):
                sampled_actions = q_algo.sample_action(episode.observations)
                values.append(
                    q_algo.predict_value(episode.observations, sampled_actions)
                )
            value = np.array(values).mean(axis=0)
            rewards = np.squeeze(episode.rewards, axis=1)
            rtg = 0
        else:
            start = max(0, idx - k + 1)
            rtg = rewards[idx] + np.maximum(rtg, value[idx + 1])  # relabel rtg
            relabelled_rewards = np.zeros_like(rewards)
            relabelled_rewards[idx] = rtg
            relabelled_rewards[start:idx] = rewards[start:idx]
            relabelled_episode = d3rlpy.dataset.components.Episode(
                observations=episode.observations,
                actions=episode.actions,
                rewards=np.expand_dims(relabelled_rewards, axis=1),
                terminated=episode.terminated,
            )
            buffer._transitions[-n - 1] = (relabelled_episode, idx)

        prev_idx = idx

    return


""" --------------------------------------------------------------------
    Fit offline RL algorithms to the given dataset.
-------------------------------------------------------------------- """


def fit_cql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    gpu: Optional[int],
    log_postfix: str,
) -> CQL:
    """
    Fit the CQL algorithm to the given dataset and environment.

    Args:
        dataset (ReplayBuffer): Dataset for the training.
        env (gym.Env): The environment instance.
        gpu (Optional[int]): The GPU device ID..
        log_postfix (str): The postfix of experiment name.

    Return:
        Trained CQL agent.
    """
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in env.spec.id:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
    ).create(device=gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=50,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_{log_postfix}",
        with_timestamp=False,
    )

    return cql


def fit_iql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    gpu: Optional[int],
    log_postfix: str,
) -> IQL:
    """
    Fit the IQL algorithm to the given dataset and environment.

    Args:
        dataset (ReplayBuffer): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (Optional[int]): The GPU device ID.
        log_postfix (str): The postfix of experiment name.

    Return:
        Trained IQL agent.
    """
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
    ).create(device=gpu)

    # workaround for learning scheduler
    iql.build_with_dataset(dataset)
    assert iql.impl
    scheduler = CosineAnnealingLR(
        iql.impl._modules.actor_optim,  # pylint: disable=protected-access
        500000,
    )

    def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
        scheduler.step()

    iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        callback=callback,
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)
        },
        experiment_name=f"IQL_{log_postfix}",
        with_timestamp=False,
    )

    return iql


def fit_dt(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    context_size: int,
    gpu: Optional[int],
    log_postfix: str,
) -> None:
    """
    Fit decisiton transformer to the given dataset and environment.

    Args:
        dataset (MDPdataset): Dataset for the training.
        env (gym.Env): The environment instance.
        context_size (int): The context size of DT.
        gpu (Optional[int]): The GPU device ID.
        log_postfix (str): The postfix of experiment name.
    """
    if "halfcheetah" in env.spec.id:
        target_return = 6000
    elif "hopper" in env.spec.id:
        target_return = 3600
    elif "walker" in env.spec.id:
        target_return = 5000
    else:
        raise ValueError("unsupported dataset")

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
            [128],
            exclude_last_activation=True,
        ),
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MultiplyRewardScaler(0.001),
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        context_size=context_size,
        num_heads=1,
        num_layers=3,
        max_timestep=1000,
    ).create(device=gpu)

    dt.fit(
        dataset,
        n_steps=100,#100000
        n_steps_per_epoch=10,#1000
        save_interval=10,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=f"QDT_{log_postfix}",
        with_timestamp=False,
    )


if __name__ == "__main__":
    main()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. hopper-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--compile", action="store_true", help="Compile model graph")
    parser.add_argument("--n_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of steps per epoch")
    parser.add_argument("--save_interval", type=int, default=1, help="Model save interval")
    parser.add_argument("--eval_gaps", type=int, default=5, help="Evaluation interval in epochs")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of evaluation trials")
    return parser.parse_args()

def resolve_env_name(dataset):
    if "halfcheetah" in dataset:
        return "HalfCheetah-v4", 6000
    elif "hopper" in dataset:
        return "Hopper-v4", 3600
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
                warmup_steps=1000
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
    ).create(device=f"cuda:{args.gpu}" if args.gpu is not None else "cpu")

    dt.fit(
        replay_buffer,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_epochs,
        save_interval=args.save_interval,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=f"DT_{args.dataset}_{args.seed}",
        n_trials=args.n_trials,
        eval_gaps=args.eval_gaps,
        logger_adapter=UnifiedFileAdapterFactory(),
    )

if __name__ == "__main__":
    main()