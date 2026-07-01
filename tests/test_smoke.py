"""Pre-cluster smoke tests — run before SLURM submission.

Each test: 1 training step on CPU, synthetic data, <5s total.
Catches import errors, shape mismatches, API breakage.

Usage:
    pytest tests/test_smoke.py -v
"""
import numpy as np
import pytest

import d3rlpy
from d3rlpy.dataset import EpisodeGenerator, create_infinite_replay_buffer
from d3rlpy.logging import NoopAdapterFactory


def _make_discrete_dataset(
    obs_dim: int = 10,
    n_actions: int = 25,
    n_steps: int = 200,
    ep_len: int = 20,
) -> d3rlpy.dataset.ReplayBuffer:
    n_episodes = n_steps // ep_len
    observations = np.random.rand(n_steps, obs_dim).astype(np.float32)
    actions = np.random.randint(n_actions, size=(n_steps, 1))
    rewards = np.random.rand(n_steps).astype(np.float32)
    terminals = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_episodes):
        terminals[(i + 1) * ep_len - 1] = 1.0
    return create_infinite_replay_buffer(
        EpisodeGenerator(observations, actions, rewards, terminals)()
    )


def _fit_one_step(algo: d3rlpy.base.LearnableBase, dataset: d3rlpy.dataset.ReplayBuffer) -> None:
    algo.fit(
        dataset,
        n_steps=1,
        n_steps_per_epoch=1,
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
    )


# ── discrete q-learning algos ──────────────────────────────────────────────

def test_smoke_discrete_bc() -> None:
    dataset = _make_discrete_dataset()
    algo = d3rlpy.algos.DiscreteBCConfig(batch_size=32).create(device="cpu")
    _fit_one_step(algo, dataset)


def test_smoke_discrete_cql() -> None:
    dataset = _make_discrete_dataset()
    algo = d3rlpy.algos.DiscreteCQLConfig(batch_size=32).create(device="cpu")
    _fit_one_step(algo, dataset)


# ── transformer algos ───────────────────────────────────────────────────────

def test_smoke_discrete_dt() -> None:
    dataset = _make_discrete_dataset()
    algo = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=4,
        context_size=5,
        num_heads=2,
        num_layers=1,
    ).create(device="cpu")
    _fit_one_step(algo, dataset)


def test_smoke_discrete_tacr() -> None:
    dataset = _make_discrete_dataset()
    algo = d3rlpy.algos.DiscreteTACRConfig(
        batch_size=4,
        context_size=5,
        num_heads=2,
        num_layers=1,
    ).create(device="cpu")
    _fit_one_step(algo, dataset)


# ── offline policy evaluation ───────────────────────────────────────────────

def test_smoke_discrete_fqe() -> None:
    dataset = _make_discrete_dataset()
    bc = d3rlpy.algos.DiscreteBCConfig(batch_size=32).create(device="cpu")
    _fit_one_step(bc, dataset)

    # FQE takes (algo, config, device) — no Config.create()
    fqe = d3rlpy.ope.DiscreteFQE(bc, d3rlpy.ope.FQEConfig(batch_size=32), device="cpu")
    fqe.fit(
        dataset,
        n_steps=1,
        n_steps_per_epoch=1,
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
    )


# ── dataset loaders ─────────────────────────────────────────────────────────

def test_smoke_get_cartpole() -> None:
    dataset, env = d3rlpy.datasets.get_cartpole()
    assert dataset.transition_count > 0
    assert env is not None


def test_smoke_imports() -> None:
    """Fail fast if any key symbol is missing."""
    from d3rlpy.algos import (
        DiscreteBC,
        DiscreteBCConfig,
        DiscreteCQL,
        DiscreteCQLConfig,
        DiscreteDecisionTransformer,
        DiscreteDecisionTransformerConfig,
        DiscreteTACR,
        DiscreteTACRConfig,
    )
    from d3rlpy.ope import DiscreteFQE, FQEConfig, FQETrajectory
    from d3rlpy.sepsis_loader import OBSERVATION_COLUMNS, get_sepsis
