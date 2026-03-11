# Master Implementation Plan
## Offline RL Research Project: DT, TACR, BC, and CQL Comparison

**Project Goal:** Compare BC, CQL-like methods, Decision Transformer (DT), and TACR across benchmark environments and clinical sepsis data with robust cluster training infrastructure.

---

## 🎯 Overview of Phases

| Phase | Focus | Duration | Status |
|-------|-------|----------|--------|
| **Phase 0** | Project audit & setup | 1-2 days | ⬜ Not Started |
| **Phase 1** | Local testing infrastructure | 2-3 days | ⬜ Not Started |
| **Phase 2** | Dataset & preprocessing | 2-4 days | ⬜ Not Started |
| **Phase 3** | Training infrastructure | 3-5 days | ⬜ Not Started |
| **Phase 4** | Evaluation infrastructure | 3-4 days | ⬜ Not Started |
| **Phase 5** | SLURM cluster setup | 2-3 days | ⬜ Not Started |
| **Phase 6** | Baseline experiments (BC, CQL) | 3-5 days | ⬜ Not Started |
| **Phase 7** | DT experiments | 3-4 days | ⬜ Not Started |
| **Phase 8** | TACR experiments | 3-4 days | ⬜ Not Started |
| **Phase 9** | Analysis & reporting | 2-3 days | ⬜ Not Started |

**Total Estimated Time:** 4-6 weeks

---

## Phase 0: Project Audit & Setup
**Goal:** Understand current state and organize workspace

### Tasks:

#### 0.1 Audit Existing Code ⬜
- [ ] Review existing algorithms in `d3rlpy/algos/`
  - [ ] Check DecisionTransformer implementation
  - [ ] Check TACR implementation (continuous & discrete)
  - [ ] Check BC implementation
  - [ ] Check CQL or similar offline RL methods
- [ ] Review existing experiment folders (`experiments/exp01-exp13`)
  - [ ] Document what each experiment does
  - [ ] Identify reusable code
- [ ] Review existing tests in `tests/`
  - [ ] Understand test structure
  - [ ] Identify coverage gaps

**Deliverable:** `PROJECT_AUDIT.md` documenting current state

#### 0.2 Create Directory Structure ⬜
```bash
d3rlpy/
├── configs/                    # All configuration files
│   ├── algorithms/
│   │   ├── bc_hopper.yaml
│   │   ├── cql_hopper.yaml
│   │   ├── dt_hopper.yaml
│   │   └── tacr_hopper.yaml
│   ├── sepsis/
│   │   ├── bc_terminal.yaml
│   │   ├── dt_terminal.yaml
│   │   └── tacr_dense.yaml
│   └── common.yaml
├── training/                   # Training scripts
│   ├── train.py               # Main training entry point
│   ├── train_benchmark.py     # Benchmark-specific
│   └── train_sepsis.py        # Sepsis-specific
├── evaluation/                 # Evaluation scripts
│   ├── evaluate_online.py     # Benchmark rollouts
│   ├── evaluate_fqe.py        # FQE for sepsis
│   └── analyze_results.py     # Result analysis
├── datasets/                   # Dataset handling
│   ├── benchmark_loader.py
│   ├── sepsis_loader.py
│   └── preprocessing.py
├── slurm/                      # Cluster submission
│   ├── train_bc.slurm
│   ├── train_cql.slurm
│   ├── train_dt.slurm
│   ├── train_tacr.slurm
│   └── batch_submit.py
└── tests/                      # Test infrastructure
    ├── import_test.py
    ├── training_step_test.py
    ├── dataset_test.py
    └── integration_test.py
```

**Deliverable:** Created directory structure

#### 0.3 Setup Environment & Dependencies ⬜
- [ ] Create/verify conda environment
- [ ] Install required packages:
  ```bash
  # Core ML
  torch
  numpy
  
  # RL & Environments
  gymnasium
  minari
  mujoco
  
  # Data & Analysis
  pandas
  scipy
  
  # Logging & Tracking
  tensorboard
  wandb (optional)
  ```
- [ ] Test imports work

**Deliverable:** Working environment, documented in `ENVIRONMENT.md`

---

## Phase 1: Local Testing Infrastructure
**Goal:** Create mandatory tests that prevent broken cluster jobs

**Critical:** Every experiment MUST pass these tests before SLURM submission.

### Tasks:

#### 1.1 Import Test ⬜
**File:** `tests/import_test.py`

**Purpose:** Verify all modules load without errors

```python
# tests/import_test.py
"""
Test that all critical modules can be imported.
Run before every cluster submission.
"""

def test_core_imports():
    """Test core library imports"""
    import torch
    import numpy as np
    import gymnasium as gym
    print("✓ Core libraries OK")

def test_algorithm_imports():
    """Test all algorithm implementations"""
    from d3rlpy.algos import BC, CQL
    from d3rlpy.algos.transformer import DecisionTransformer, TACR
    from d3rlpy.algos.transformer import DiscreteDecisionTransformer, DiscreteTACR
    print("✓ Algorithm imports OK")

def test_dataset_imports():
    """Test dataset and preprocessing"""
    from d3rlpy.dataset import ReplayBuffer
    # Add custom dataset imports
    print("✓ Dataset imports OK")

def main():
    print("="*50)
    print("IMPORT TEST")
    print("="*50)
    test_core_imports()
    test_algorithm_imports()
    test_dataset_imports()
    print("\n✅ All imports successful!")
    
if __name__ == "__main__":
    main()
```

**Run:** `python -m tests.import_test`

**Deliverable:** Working import test

#### 1.2 Single Batch Training Test ⬜
**File:** `tests/training_step_test.py`

**Purpose:** Verify one forward + backward pass works

```python
# tests/training_step_test.py
"""
Test that a single training step completes successfully.
This catches most configuration and implementation errors.
"""

def test_bc_step():
    """Test BC can do one training step"""
    pass

def test_cql_step():
    """Test CQL can do one training step"""
    pass

def test_dt_step():
    """Test DT can do one training step"""
    pass

def test_tacr_step():
    """Test TACR can do one training step"""
    pass

def main():
    print("="*50)
    print("SINGLE BATCH TRAINING TEST")
    print("="*50)
    test_bc_step()
    test_cql_step()
    test_dt_step()
    test_tacr_step()
    print("\n✅ All training steps successful!")

if __name__ == "__main__":
    main()
```

**Run:** `python -m tests.training_step_test`

**Deliverable:** Working training step test

#### 1.3 Small Dataset Test ⬜
**File:** `tests/dataset_test.py`

**Purpose:** Verify dataloader works with small subset

```python
# tests/dataset_test.py
"""
Test dataset loading and preprocessing with small data.
"""

def test_benchmark_dataset():
    """Test loading 10 benchmark trajectories"""
    pass

def test_sepsis_dataset():
    """Test loading 10 sepsis patient trajectories"""
    pass

def test_dataloader():
    """Test PyTorch DataLoader functionality"""
    pass

def main():
    print("="*50)
    print("DATASET TEST")
    print("="*50)
    test_benchmark_dataset()
    test_sepsis_dataset()
    test_dataloader()
    print("\n✅ All dataset tests passed!")

if __name__ == "__main__":
    main()
```

**Run:** `python -m tests.dataset_test`

**Deliverable:** Working dataset test

#### 1.4 Environment Rollout Test ⬜
**File:** `tests/rollout_test.py`

**Purpose:** Verify policy can interact with environment

```python
# tests/rollout_test.py
"""
Test that trained policies can perform environment rollouts.
Only for benchmark environments (sepsis has no environment).
"""

def test_random_policy_rollout():
    """Test random policy rollout"""
    pass

def test_bc_rollout():
    """Test BC policy rollout"""
    pass

def test_dt_rollout():
    """Test DT policy rollout with RTG"""
    pass

def main():
    print("="*50)
    print("ROLLOUT TEST")
    print("="*50)
    test_random_policy_rollout()
    test_bc_rollout()
    test_dt_rollout()
    print("\n✅ All rollout tests passed!")

if __name__ == "__main__":
    main()
```

**Run:** `python -m tests.rollout_test`

**Deliverable:** Working rollout test

#### 1.5 Pre-Cluster Validation Script ⬜
**File:** `tests/run_all_tests.sh`

**Purpose:** Run all tests before SLURM submission

```bash
#!/bin/bash
# tests/run_all_tests.sh

echo "=================================="
echo "PRE-CLUSTER VALIDATION"
echo "=================================="

echo ""
echo "Step 1: Import Test"
python -m tests.import_test || exit 1

echo ""
echo "Step 2: Dataset Test"
python -m tests.dataset_test || exit 1

echo ""
echo "Step 3: Training Step Test"
python -m tests.training_step_test || exit 1

echo ""
echo "Step 4: Rollout Test"
python -m tests.rollout_test || exit 1

echo ""
echo "=================================="
echo "✅ ALL TESTS PASSED!"
echo "Ready for cluster submission."
echo "=================================="
```

**Run:** `bash tests/run_all_tests.sh`

**Deliverable:** Master validation script

---

## Phase 2: Dataset & Preprocessing
**Goal:** Reliable data loading for both domains

### Tasks:

#### 2.1 Benchmark Dataset Loader ⬜
**File:** `datasets/benchmark_loader.py`

**Features:**
- Load Minari datasets (Hopper, Walker2D, HalfCheetah)
- Continuous states & actions
- Trajectory segmentation
- Train/val split

**Deliverable:** `BenchmarkDataset` class

#### 2.2 Sepsis Dataset Loader ⬜
**File:** `datasets/sepsis_loader.py`

**Features:**
- Load MIMIC-IV patient trajectories
- Continuous clinical features (normalized)
- Discrete actions (fluids_bin × vasopressor_bin)
- Two reward modes:
  - Terminal: +1 survive, -1 death, 0 otherwise
  - Dense: SOFA-based with optional terminal bonus
- Train/val split

**Deliverable:** `SepsisDataset` class

#### 2.3 Preprocessing Pipeline ⬜
**File:** `datasets/preprocessing.py`

**Features:**
- State normalization
- Action discretization (sepsis)
- Trajectory truncation/padding
- Context window extraction (for DT/TACR)

**Deliverable:** Preprocessing utilities

#### 2.4 Dataset Configuration ⬜
**Files:** `configs/datasets/*.yaml`

**Example:**
```yaml
# configs/datasets/hopper_medium.yaml
dataset:
  name: "hopper-medium-v2"
  source: "minari"
  train_ratio: 0.9
  max_trajectories: null
  context_length: 20

# configs/datasets/sepsis_terminal.yaml
dataset:
  name: "mimic_sepsis"
  source: "local"
  path: "data/sepsis/processed.pkl"
  reward_mode: "terminal"  # or "dense"
  train_ratio: 0.9
  max_trajectories: null
  context_length: 50
  action_space:
    type: "discrete"
    fluids_bins: 5
    vasopressor_bins: 5
```

**Deliverable:** Dataset configs

---

## Phase 3: Training Infrastructure
**Goal:** Unified training system with config files

### Tasks:

#### 3.1 Main Training Script ⬜
**File:** `training/train.py`

**Features:**
- Load config from YAML
- Initialize algorithm (BC/CQL/DT/TACR)
- Load dataset (benchmark/sepsis)
- Training loop with logging
- Checkpoint saving
- Support for both continuous and discrete action spaces

**Usage:**
```bash
python training/train.py --config configs/algorithms/dt_hopper.yaml
```

**Deliverable:** Universal training script

#### 3.2 Algorithm Configurations ⬜
**Files:** `configs/algorithms/*.yaml`

**Example:**
```yaml
# configs/algorithms/dt_hopper.yaml
experiment:
  name: "dt_hopper_medium"
  seed: 42
  device: "cuda"

dataset:
  name: "hopper-medium-v2"
  source: "minari"
  context_length: 20

algorithm:
  type: "DecisionTransformer"
  params:
    embed_dim: 128
    num_layers: 3
    num_heads: 1
    context_size: 20
    max_timestep: 1000
    learning_rate: 0.0001
    batch_size: 64
    
training:
  epochs: 100
  eval_interval: 10
  save_interval: 10
  max_steps: null

evaluation:
  type: "online"  # online rollout
  episodes: 100
  target_return: 3600  # for DT/TACR
```

```yaml
# configs/algorithms/tacr_sepsis_terminal.yaml
experiment:
  name: "tacr_sepsis_terminal"
  seed: 42
  device: "cuda"

dataset:
  name: "mimic_sepsis"
  source: "local"
  path: "data/sepsis/terminal_reward.pkl"
  context_length: 50
  reward_mode: "terminal"

algorithm:
  type: "DiscreteTACR"
  params:
    embed_dim: 256
    num_layers: 4
    num_heads: 4
    context_size: 50
    max_timestep: 100
    learning_rate: 0.0001
    batch_size: 128
    actor_coef: 1.0
    critic_coef: 1.0
    kl_coef: 0.1

training:
  epochs: 200
  eval_interval: 20
  save_interval: 20

evaluation:
  type: "fqe"  # Fitted Q Evaluation
  target_return: 1.0
```

**Deliverable:** Config files for all algorithms

#### 3.3 Training Utilities ⬜
**File:** `training/utils.py`

**Features:**
- Config loading/validation
- Model initialization
- Optimizer setup
- Learning rate scheduling
- Logging setup (TensorBoard/WandB)
- Checkpoint management

**Deliverable:** Training utility functions

#### 3.4 Debug Mode ⬜
**Feature:** `--debug` flag for rapid testing

**Behavior:**
```bash
python training/train.py --config configs/algorithms/dt_hopper.yaml --debug
```

**Changes in debug mode:**
- Max 10 trajectories
- 1 epoch only
- No checkpointing
- Verbose logging

**Deliverable:** Debug mode implementation

---

## Phase 4: Evaluation Infrastructure
**Goal:** Proper evaluation for both domains

### Tasks:

#### 4.1 Online Evaluation (Benchmarks) ⬜
**File:** `evaluation/evaluate_online.py`

**Features:**
- Load trained policy
- Perform environment rollouts
- For DT/TACR: RTG-based evaluation starting at t=0
  ```python
  rtg = target_return
  for t in range(max_steps):
      action = policy(state, rtg, t)
      state, reward, done = env.step(action)
      rtg = rtg - reward  # Update RTG
  ```
- Compute metrics:
  - Mean episodic return
  - Std return
  - Normalized score
- Save results

**Usage:**
```bash
python evaluation/evaluate_online.py \
  --checkpoint models/dt_hopper/checkpoint_100.pt \
  --config configs/algorithms/dt_hopper.yaml \
  --episodes 100 \
  --target-return 3600
```

**Deliverable:** Online evaluation script

#### 4.2 Offline Evaluation (FQE) ⬜
**File:** `evaluation/evaluate_fqe.py`

**Features:**
- Load trained policy
- Load test dataset
- Train FQE Q-network
- Estimate policy value
- For DT/TACR: start at t=0 with target RTG
- Save results

**Usage:**
```bash
python evaluation/evaluate_fqe.py \
  --checkpoint models/tacr_sepsis/checkpoint_200.pt \
  --config configs/algorithms/tacr_sepsis_terminal.yaml \
  --target-return 1.0
```

**Deliverable:** FQE evaluation script

#### 4.3 Secondary Sepsis Metrics ⬜
**File:** `evaluation/sepsis_analysis.py`

**Features:**
- Clinician action agreement
- Subgroup analysis (e.g., by severity)
- Action distribution plausibility
- Survival rate stratification

**Deliverable:** Sepsis-specific analysis

#### 4.4 Results Analysis ⬜
**File:** `evaluation/analyze_results.py`

**Features:**
- Aggregate results across seeds
- Generate comparison tables
- Create plots
- Statistical significance tests

**Deliverable:** Analysis tools

---

## Phase 5: SLURM Cluster Setup
**Goal:** Reliable cluster training with validation

### Tasks:

#### 5.1 SLURM Job Templates ⬜
**Files:** `slurm/*.slurm`

**Example:**
```bash
#!/bin/bash
# slurm/train_dt.slurm

#SBATCH --job-name=dt_hopper
#SBATCH --output=logs/dt_hopper_%j.out
#SBATCH --error=logs/dt_hopper_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load environment
source ~/.bashrc
conda activate rl_env

# Navigate to project
cd /path/to/d3rlpy_new/d3rlpy

# Run pre-cluster validation
echo "Running validation tests..."
bash tests/run_all_tests.sh
if [ $? -ne 0 ]; then
    echo "❌ Validation failed! Job cancelled."
    exit 1
fi

# Run training
echo "Starting training..."
python training/train.py --config configs/algorithms/dt_hopper.yaml
```

**Deliverable:** SLURM scripts for each algorithm

#### 5.2 Batch Submission Script ⬜
**File:** `slurm/batch_submit.py`

**Features:**
- Submit multiple jobs with different configs
- Different seeds
- Dependency management

**Usage:**
```bash
python slurm/batch_submit.py \
  --algorithm dt \
  --environments hopper walker2d halfcheetah \
  --seeds 42 43 44 45 46 \
  --dry-run  # Preview jobs without submitting
```

**Deliverable:** Batch submission tool

#### 5.3 Job Monitoring ⬜
**File:** `slurm/monitor_jobs.py`

**Features:**
- Check job status
- Parse logs for errors
- Alert on failures

**Deliverable:** Monitoring tool

---

## Phase 6: Baseline Experiments (BC, CQL)
**Goal:** Establish baseline performance

### Tasks:

#### 6.1 Behavior Cloning Experiments ⬜

**Benchmarks:**
- [ ] Hopper-medium-v2 (5 seeds)
- [ ] Walker2D-medium-v2 (5 seeds)
- [ ] HalfCheetah-medium-v2 (5 seeds)

**Sepsis:**
- [ ] Terminal reward (5 seeds)
- [ ] Dense reward (5 seeds)

**Deliverable:** BC baseline results

#### 6.2 CQL Experiments ⬜

**Benchmarks:**
- [ ] Hopper-medium-v2 (5 seeds)
- [ ] Walker2D-medium-v2 (5 seeds)
- [ ] HalfCheetah-medium-v2 (5 seeds)

**Sepsis:**
- [ ] Terminal reward (5 seeds)
- [ ] Dense reward (5 seeds)

**Deliverable:** CQL baseline results

---

## Phase 7: Decision Transformer Experiments
**Goal:** Evaluate sequence modeling approach

### Tasks:

#### 7.1 DT Benchmark Experiments ⬜
- [ ] Hopper (5 seeds)
- [ ] Walker2D (5 seeds)
- [ ] HalfCheetah (5 seeds)
- [ ] Hyperparameter sweep on one environment

**Deliverable:** DT benchmark results

#### 7.2 DT Sepsis Experiments ⬜
- [ ] Terminal reward (5 seeds)
- [ ] Dense reward (5 seeds)
- [ ] RTG sensitivity analysis

**Deliverable:** DT sepsis results

---

## Phase 8: TACR Experiments
**Goal:** Evaluate proposed method

### Tasks:

#### 8.1 TACR Benchmark Experiments ⬜
- [ ] Hopper (5 seeds)
- [ ] Walker2D (5 seeds)
- [ ] HalfCheetah (5 seeds)
- [ ] Ablation: actor/critic/KL coefficients

**Deliverable:** TACR benchmark results

#### 8.2 TACR Sepsis Experiments ⬜
- [ ] Terminal reward (5 seeds)
- [ ] Dense reward (5 seeds)
- [ ] Comparison with DT

**Deliverable:** TACR sepsis results

---

## Phase 9: Analysis & Reporting
**Goal:** Comprehensive result analysis

### Tasks:

#### 9.1 Result Aggregation ⬜
- [ ] Collect all experiment results
- [ ] Compute mean ± std across seeds
- [ ] Statistical significance tests

**Deliverable:** Aggregated results

#### 9.2 Comparative Analysis ⬜
- [ ] BC vs CQL vs DT vs TACR tables
- [ ] Learning curves
- [ ] Success rate analysis (sepsis)

**Deliverable:** Comparison tables and plots

#### 9.3 Ablation Studies ⬜
- [ ] RTG sensitivity (DT/TACR)
- [ ] Context length sensitivity
- [ ] Reward mode comparison (sepsis)

**Deliverable:** Ablation results

#### 9.4 Final Report ⬜
- [ ] Write methods section
- [ ] Create result figures
- [ ] Discussion of findings

**Deliverable:** Research report/paper

---

## 📋 Quick Start Checklist

Before starting any phase, ensure:

- [ ] Git repository is clean
- [ ] Conda environment is activated
- [ ] All dependencies are installed
- [ ] You understand the phase goals

Before submitting cluster jobs:

- [ ] Code passes `tests/run_all_tests.sh`
- [ ] Config files are validated
- [ ] Checkpoint directory exists
- [ ] Logs directory exists

---

## 🚨 Common Pitfalls to Avoid

1. **Submitting untested code to cluster** → Always run local tests first
2. **Hard-coding hyperparameters** → Use config files
3. **Ignoring discrete vs continuous actions** → Check algorithm variant
4. **Wrong RTG initialization** → Always start at t=0 for DT/TACR
5. **Mixed reward modes** → Be explicit about terminal vs dense
6. **No logging** → Always log training progress
7. **No checkpointing** → Save models regularly
8. **No seed control** → Set seeds for reproducibility

---

## 📊 Expected Timeline

**Conservative estimate (with buffer):** 6 weeks

**Aggressive estimate (ideal conditions):** 4 weeks

**Critical path:**
1. Testing infrastructure (must complete first)
2. Training infrastructure (blocks experiments)
3. Evaluation infrastructure (needed for results)
4. Experiments can run in parallel after infrastructure is ready

---

## 🎓 Next Steps

**Recommend starting with:**

1. **Phase 0.1** - Audit existing code to understand what's already implemented
2. **Phase 1** - Build testing infrastructure (critical path)
3. **Phase 2.1** - Get one dataset working end-to-end
4. **Phase 3.1** - Get one algorithm training successfully
5. Then expand systematically

**Would you like to start with Phase 0.1 (Project Audit)?**
