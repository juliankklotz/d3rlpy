# Implementation Plan
**Updated:** 2026-06-30

---

## Thesis Goal

Compare **DiscreteBC / DiscreteCQL / DiscreteDecisionTransformer / DiscreteTACR** (thesis contribution) across three discrete-action offline RL domains: CartPole, Atari Pong, Sepsis.

---

## Benchmarks

| Domain | Dataset source | Actions | Eval |
|--------|---------------|---------|------|
| CartPole-v1 | `get_cartpole()` in [datasets.py](d3rlpy/datasets.py) | 2 | `EnvironmentEvaluator`, target 200 |
| Atari Pong | `get_minari('atari/pong/expert-v0')` | 6 | `EnvironmentEvaluator`, target 20 |
| Sepsis (MIMIC-IV) | [sepsis_loader.py](d3rlpy/sepsis_loader.py) | 25 (5 fluids × 5 vasopressors) | FQE only — no simulator |

---

## What's Done

### Algorithms — all in [d3rlpy/algos/](d3rlpy/algos/)

| Class | File | Status |
|-------|------|--------|
| `DiscreteBC` | [algos/qlearning/bc.py](d3rlpy/algos/qlearning/bc.py) | ✅ |
| `DiscreteCQL` | [algos/qlearning/cql.py](d3rlpy/algos/qlearning/cql.py) | ✅ |
| `DiscreteDecisionTransformer` | [algos/transformer/decision_transformer.py](d3rlpy/algos/transformer/decision_transformer.py) | ✅ |
| `DiscreteTACR` | [algos/transformer/tacr.py](d3rlpy/algos/transformer/tacr.py) | ✅ |

### Offline Evaluation — [d3rlpy/ope/fqe.py](d3rlpy/ope/fqe.py)

- `DiscreteFQE` — for BC/CQL
- `FQETrajectory` — for DT/TACR (handles trajectory context)
- `DTConstantRTGforFQE` — RTG wrapper for eval

### Online Evaluation

- `EnvironmentEvaluator` — [metrics/evaluators.py](d3rlpy/metrics/evaluators.py)
- `DiscountedRTGWrapper` — [algos/transformer/base.py](d3rlpy/algos/transformer/base.py)

### Existing Tests

- TACR: [tests/algos/transformer/test_tacr.py](tests/algos/transformer/test_tacr.py)
- DT: [tests/algos/transformer/test_decision_transformer.py](tests/algos/transformer/test_decision_transformer.py)
- FQE: [tests/ope/test_fqe.py](tests/ope/test_fqe.py)
- BC/CQL: [tests/algos/qlearning/](tests/algos/qlearning/)

### Working Experiments

| Exp | What | Key file |
|-----|------|---------|
| [exp07_discrete_DT/](experiments/exp07_discrete_DT/) | DiscreteDT on CartPole + Pong | notebooks |
| [exp08_discrete_TACR/](experiments/exp08_discrete_TACR/) | DiscreteTACR on CartPole + Pong + Hopper; SLURM tuning | [run_discrete_TACR.py](experiments/exp08_discrete_TACR/run_discrete_TACR.py) |
| [exp09_online_algo_sepsis/](experiments/exp09_online_algo_sepsis/) | Online algo on sepsis env | [gym-sepsis/](experiments/exp09_online_algo_sepsis/gym-sepsis/) |
| [exp11_mimic_exploration/](experiments/exp11_mimic_exploration/) | MIMIC-IV data exploration | [mimic_exploration.ipynb](experiments/exp11_mimic_exploration/mimic_exploration.ipynb) |
| [exp13_FQE_with_trajectories/](experiments/exp13_FQE_with_trajectories/) | Trajectory-based FQE | [eval_cql_with_trajectories.ipynb](experiments/exp13_FQE_with_trajectories/eval_cql_with_trajectories.ipynb) |

### Reproduction scripts — [reproductions/offline/](reproductions/offline/)

- `decision_transformer.py`, `tacr.py`, `cql.py`, `discrete_cql.py`, `discrete_decision_transformer.py`
- Already have argparse CLI, `EnvironmentEvaluator`, checkpoint saving — good templates for unified script

### SLURM

- Working template: [experiments/exp08_discrete_TACR/discrete_TACR_hyperparameter_tuning_two.sh](experiments/exp08_discrete_TACR/discrete_TACR_hyperparameter_tuning_two.sh)
- Data paths already configured (`D3RLPY_DATASETS_PATH`, `MINARI_DATASETS_PATH`)

---

## What Needs to Be Done

### Priority 1 — Sepsis integration (Days 1–3)

**Gap:** [sepsis_loader.py](d3rlpy/sepsis_loader.py) exists but isn't producing a `ReplayBuffer`/`MDPDataset` compatible with the training loop.

Tasks:
- [ ] Wrap MIMIC trajectories → `Episode` objects ([dataset/components.py](d3rlpy/dataset/components.py))
- [ ] Implement terminal reward (`+1` survive / `-1` die)
- [ ] Implement dense reward (SOFA-based, see [exp11 notebook](experiments/exp11_mimic_exploration/mimic_exploration.ipynb))
- [ ] Smoke test: `DiscreteBC.fit(dataset, n_steps=100)` passes

### Priority 2 — Unified training script (Days 3–5)

**Gap:** Experiments scattered in notebooks; [exp08/run_discrete_TACR.py](experiments/exp08_discrete_TACR/run_discrete_TACR.py) is the closest to a real script but TACR-only.

Tasks:
- [ ] Create `training/train.py`: algo + dataset CLI args, `--debug` flag for 10-step smoke test
- [ ] Base it on [reproductions/offline/discrete_decision_transformer.py](reproductions/offline/discrete_decision_transformer.py) pattern (already handles evaluators + checkpointing)
- [ ] Cover all 4 algorithms + 3 datasets including sepsis

### Priority 3 — Pre-cluster validation (Days 5–6)

**Gap:** No quick smoke test before SLURM submission.

Tasks:
- [ ] `tests/integration/test_smoke.py` — one training step per algo per dataset (10 steps, CPU)
- [ ] Shell wrapper `tests/pre_cluster_check.sh` that runs it and exits non-zero on failure
- [ ] Add call to that wrapper at top of SLURM job script

### Priority 4 — SLURM standardization (Day 7)

**Gap:** Only exp08 has a working template; needs generalizing.

Tasks:
- [ ] `slurm/train.slurm` — parameterized template (`{ALGO}`, `{DATASET}`, `{SEED}`)
- [ ] `slurm/submit_all.sh` — loop over matrix, call `sbatch` with env-var substitution
- [ ] Wire in `pre_cluster_check.sh`

### Priority 5 — Evaluation automation (Days 8–10)

Tasks:
- [ ] `evaluation/eval_online.py` — runs `EnvironmentEvaluator` on a checkpoint, writes JSON
- [ ] `evaluation/eval_fqe.py` — runs `DiscreteFQE` / `FQETrajectory` on a checkpoint, writes JSON
- [ ] `evaluation/aggregate.py` — reads JSON files, outputs mean±std table per algo×dataset

---

## Experiment Matrix

4 algorithms × 3 datasets × 3 seeds = **36 runs**

| | CartPole | Pong | Sepsis |
|--|--|--|--|
| DiscreteBC | 3 seeds | 3 seeds | 3 seeds |
| DiscreteCQL | 3 seeds | 3 seeds | 3 seeds |
| DiscreteDT | 3 seeds | 3 seeds | 3 seeds |
| DiscreteTACR | ✅ done | ✅ done | 3 seeds |

CartPole + Pong: online eval. Sepsis: FQE only.

---

## Timeline

| Days | Focus | Deliverable |
|------|-------|-------------|
| 1–3 | Sepsis loader | `DiscreteBC.fit(sepsis_dataset)` works |
| 3–5 | `training/train.py` | single command runs any algo × dataset |
| 5–6 | Pre-cluster tests | `bash tests/pre_cluster_check.sh` exits 0 |
| 7 | SLURM templates | one-command batch submission |
| 8–10 | Eval scripts | automated result collection |
| 11–17 | Submit + monitor | all 36 runs on cluster |
| 18–21 | Collect results | eval scripts produce tables |
| 22–25 | Analysis + writing | plots, significance tests, thesis text |

**Buffer:** 5 days for cluster queue, failed jobs, dataset surprises.

---

## Risk Factors

- **Sepsis data quality** — exp11 exploration shows missing/inconsistent columns; budget 1 extra day
- **Pong training time** — Atari is slower; may need 24h wall time per job
- **FQE instability for DT/TACR** — trajectory-based FQE still experimental; fall back to `DiscreteFQE` with fixed context if needed
