import numpy as np

from ..interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from ..types import GymEnv

__all__ = [
    "evaluate_qlearning_with_environment",
    "evaluate_transformer_with_environment",
    "evaluate_var_rtg_transformer_with_environment",
]


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(observation, np.ndarray):
                    observation = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation = [
                        np.expand_dims(o, axis=0) for o in observation
                    ]
                else:
                    raise ValueError(
                        f"Unsupported observation type: {type(observation)}"
                    )
                action = algo.predict(observation)[0]

            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))


def evaluate_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
) -> dict:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        if env.spec.id.startswith("ALE"):
            observation = observation.transpose(2, 0, 1)

        episode_reward = 0.0
        step_count = 0
        while True:
            # take action
            action = algo.predict(observation, reward)

            observation, _reward, done, truncated, _ = env.step(action)
            if env.spec.id.startswith("ALE"):
                observation = observation.transpose(2, 0, 1)
            reward = float(_reward)
            episode_reward += reward
            step_count += 1

            if done or truncated or step_count >= algo._max_timestep:
                break
        episode_rewards.append(episode_reward)
    output = {
        "episode_mean_reward": float(np.mean(episode_rewards)),
        "episode_median_reward": float(np.median(episode_rewards)),
        "episode_std_reward": float(np.std(episode_rewards)),
        "episode_min_reward": float(np.min(episode_rewards)),
        "episode_max_reward": float(np.max(episode_rewards)),
        "episode_count": n_trials,
    }
    return output



def evaluate_var_rtg_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: GymEnv,
    q_algo: QLearningAlgoProtocol,
    n_trials: int = 10,
    num_action_samples: int = 100,
    ambition_parameter: float = 0.1,
) -> dict:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0
        values = []
        observation_batch = observation[None, ...]

        for _ in range(num_action_samples):
            sampled_actions_batch = q_algo.sample_action(observation_batch)
            values_batch = q_algo.predict_value(observation_batch,sampled_actions_batch)
            values.append(values_batch.item())
        best_value = float(np.max(values))
        target_return = min(best_value * (1+ambition_parameter),algo._target_return)
        #algo._target_return = best_value
        algo.__setattr__("_target_return", target_return)
        while True:
            # take action
            action = algo.predict(observation, reward)

            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    output = {
        "episode_mean_reward": float(np.mean(episode_rewards)),
        "episode_median_reward": float(np.median(episode_rewards)),
        "episode_std_reward": float(np.std(episode_rewards)),
        "episode_min_reward": float(np.min(episode_rewards)),
        "episode_max_reward": float(np.max(episode_rewards)),
        "episode_count": n_trials,
    }
    return output