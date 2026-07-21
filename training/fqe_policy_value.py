#!/usr/bin/env python3
"""Correct FQE policy-value metric: V_hat(pi) at initial states under the policy.

Implements the thesis eval objective (section 5.5.2):

    V_hat(pi) = (1 / N_test) * sum_j  Q_hat^pi(s_0^j, pi(s_0^j))

evaluated at the INITIAL state of each held-out test trajectory, with the action
taken from the POLICY pi (not the logged clinician action). This is distinct
from — and the correct replacement for — the earlier train_sepsis.run_fqe metric,
which averaged Q_hat over ALL transitions using the logged clinician actions
(that measures the critic's opinion of the clinician, not the value of pi).

For consistency with how the FQE critic was fit (expected-SARSA bootstrap under
pi's action distribution, see DiscreteFQETrajectory), the per-trajectory value is
the expected Q under the policy at s_0:

    v_j = sum_a pi(a | s_0^j) * Q_hat(s_0^j, a)

  - Q-learning policies (BC, CQL): pi is greedy -> point mass at argmax_a Q,
    so v_j = max_a Q_hat(s_0, a) is NOT used; instead we use the algo's own
    action probabilities where available, else greedy.
  - DT/TACR: pi(a|s_0) is the softmax over actions at the fixed target RTG.

Returns per-trajectory values so the caller can aggregate (mean, std, CI) across
trajectories, folds, and seeds.
"""
from typing import Optional

import numpy as np


def fqe_policy_value_per_trajectory(
    fqe,
    algo,
    test_dataset,
    algo_name: str,
    target_return: Optional[float] = None,
) -> np.ndarray:
    """Return one FQE value estimate per test trajectory, evaluated at s_0.

    Args:
        fqe: a FITTED FQE estimator (DiscreteFQE or DiscreteFQETrajectory) whose
            predict_value(obs, action) returns Q_hat(obs, action).
        algo: the trained policy being evaluated (used to get pi(a|s_0)).
        test_dataset: ReplayBuffer of held-out test trajectories.
        algo_name: one of discrete_bc, discrete_cql, discrete_dt, discrete_tacr.
        target_return: fixed RTG for DT/TACR (ignored for BC/CQL).

    Returns:
        np.ndarray of shape (n_test_trajectories,): v_j for each trajectory.
    """
    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")

    # action_size from the dataset
    action_size = test_dataset.dataset_info.action_size

    values = []
    for episode in test_dataset.episodes:
        obs = episode.observations
        if not isinstance(obs, np.ndarray) or len(obs) == 0:
            continue
        s0 = obs[0:1]  # (1, obs_dim), keep batch dim

        # pi(a | s_0): distribution over the discrete action set
        probs = _policy_action_probs(
            algo, s0, action_size, is_transformer, target_return
        )  # (action_size,)

        # Q_hat(s_0, a) for every action a, via the fitted FQE critic
        q_s0 = np.array(
            [
                float(fqe.predict_value(s0, np.array([a], dtype=np.int64))[0])
                for a in range(action_size)
            ]
        )  # (action_size,)

        v_j = float(np.dot(probs, q_s0))  # expected Q under pi at s_0
        values.append(v_j)

    return np.array(values, dtype=np.float64)


def _policy_action_probs(algo, s0, action_size, is_transformer, target_return):
    """Return pi(a | s_0) as a length-action_size probability vector."""
    import torch

    if is_transformer:
        # DT/TACR: softmax over actions at the fixed target RTG, from the actor.
        # Query the transformer directly for the action distribution at s_0.
        # A single-step context (the initial state) with the target RTG.
        impl = algo.impl
        transformer = impl._modules.transformer
        device = impl.device

        obs_t = torch.as_tensor(s0, dtype=torch.float32, device=device).unsqueeze(1)  # (1,1,obs)
        if algo.config.observation_scaler:
            # scaler expects (B, obs); apply on the flattened view
            flat = obs_t.reshape(-1, *obs_t.shape[2:])
            flat = algo.config.observation_scaler.transform(flat)
            obs_t = flat.reshape(1, 1, *obs_t.shape[2:])
        act_t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
        rtg_val = float(target_return) if target_return is not None else 0.0
        rtg_t = torch.full((1, 1, 1), rtg_val, dtype=torch.float32, device=device)
        steps_t = torch.zeros((1, 1), dtype=torch.int64, device=device)
        pad_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

        with torch.no_grad():
            probs, _ = transformer(obs_t, act_t, rtg_t, steps_t, pad_t)
        probs = probs.reshape(-1, action_size)[0]
        return probs.detach().cpu().numpy().astype(np.float64)

    # Q-learning (BC, CQL): greedy policy -> one-hot at argmax_a Q(s_0, a).
    # Use the algo's own best action (respects its own scalers/head).
    best_a = int(algo.predict(s0)[0])
    p = np.zeros(action_size, dtype=np.float64)
    p[best_a] = 1.0
    return p
