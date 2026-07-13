# Testing Checklist for d3rlpy Thesis

Run these tests before cluster submission to catch bugs early.

## Local Validation (CPU, ~10 min)

### 1. Sepsis Data Loading
```bash
cd d3rlpy
export SEPSIS_DATA_DIR="./data"
python -c "
from d3rlpy.sepsis_loader import get_sepsis
import numpy as np

for mode in ['terminal', 'dense', 'mixed']:
    ds = get_sepsis(reward_mode=mode)
    ep = ds.episodes[0]
    rewards = ep.rewards
    nan_count = np.isnan(rewards).sum()
    if nan_count == 0:
        r_min, r_max = np.min(rewards), np.max(rewards)
        print(f'✓ {mode}: no NaN, range=[{r_min:.2f}, {r_max:.2f}]')
    else:
        print(f'✗ {mode}: NaN count={nan_count}')
"
```
**Expected**: All three show `✓` with valid ranges. No NaN.

### 2. Smoke Tests
```bash
pytest tests/test_smoke.py -v
```
**Expected**: All 7 tests pass in ~5s.

### 3. CartPole Training (1 step, 30s)
```bash
python training/train_benchmarks.py \
    --algo discrete_bc \
    --dataset cartpole \
    --seed 0 \
    --device cpu \
    --n_steps 1
```
**Expected**: Completes without error, creates d3rlpy_logs/ directory.

### 4. Sepsis Terminal Training (1 step, 60s)
```bash
python training/train_sepsis.py \
    --algo discrete_bc \
    --reward_mode terminal \
    --seed 0 \
    --device cpu \
    --n_steps 1 \
    --skip_fqe
```
**Expected**: Completes, model saved to d3rlpy_logs/.

### 5. Sepsis Mixed Reward Training (1 step, 60s)
```bash
python training/train_sepsis.py \
    --algo discrete_bc \
    --reward_mode mixed \
    --seed 0 \
    --device cpu \
    --n_steps 1 \
    --skip_fqe
```
**Expected**: Completes without NaN warnings.

### 6. Cluster Validation
```bash
bash scripts/pre_cluster_check
```
**Expected**: All smoke tests pass, no import errors.

## Pre-Cluster Verification

Before running `submit_all_jobs.sh`:

1. ✓ All tests above pass
2. ✓ No NaN in rewards (test #1)
3. ✓ $SEPSIS_DATA_DIR is set and contains CSVs
4. ✓ Git is clean: `git status` shows only untracked d3rlpy_logs/
5. ✓ SLURM template updated with correct hardcoded Python path

## Common Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: d3rlpy` | Env not sourced | `source activate d3rlpy_dev_final_py310` |
| `NaN in episode.rewards` | Sepsis reward computation | Check sepsis_loader.py fillna() logic |
| `SEPSIS_DATA_DIR not set` | Env var missing | `export SEPSIS_DATA_DIR=/path/to/data` |
| `FileNotFoundError: mimic_dataset.csv` | Wrong data_dir | Verify files exist in $SEPSIS_DATA_DIR |
| `atari-py build error` | d4rl-atari issue | Use `--dataset pong_minari` instead |

## Running Full Suite

Once all above pass:

```bash
cd d3rlpy
bash slurm/submit_all_jobs.sh  # Submits 48 jobs
squeue -u jklotz               # Monitor
```

Expected wall time: CartPole 30min, Pong 3h, Sepsis-terminal 4h, Sepsis-mixed 4h per job.
With 3 seeds × 4 algos = parallelism depends on cluster queue limits.
