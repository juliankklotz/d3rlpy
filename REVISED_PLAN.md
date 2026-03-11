# Revised 25-Day Implementation Plan
**Based on Actual Experiments**  
**Date:** March 7, 2026

---

## 🎯 Project Context

**Repository:** Fork of d3rlpy (offline RL library)  
**Your Contribution:** TACR implementation + Sepsis application  
**Goal:** Compare algorithms across your tested benchmarks + sepsis domain

---

## 📊 Benchmark Environments (ALL DISCRETE ACTIONS)

### 1. **CartPole-v1** (Simple Discrete Control)
- **Source:** `d3rlpy.datasets.get_cartpole()`
- **Action Space:** Discrete (2 actions: left/right)
- **State Space:** Continuous (4D: position, velocity, angle, angular velocity)
- **Target Return:** 200
- **Tested In:** exp07, exp08
- **Status:** ✅ Working with DiscreteDT, DiscreteTACR

### 2. **Atari Pong** (Visual, Discrete)
- **Source:** `d3rlpy.datasets.get_minari('atari/pong/expert-v0')`
- **Action Space:** Discrete (6 actions)
- **State Space:** Continuous (pixel frames or features)
- **Target Return:** 20
- **Tested In:** exp07, exp08
- **Status:** ✅ Working with DiscreteDT, DiscreteTACR

### 3. **Sepsis Treatment** (Clinical Decision Making)
- **Source:** MIMIC-IV data (exp11)
- **Action Space:** Discrete (25 actions: 5 fluids × 5 vasopressors)
- **State Space:** Continuous (clinical features: vitals, labs, SOFA score)
- **Evaluation:** Offline only (FQE) - no environment available
- **Reward Modes:** Terminal (+1 survive, -1 death) OR Dense (SOFA-based)
- **Status:** ⚠️ Data exists, needs integration

---

## ✅ Simplified Algorithm Set (Discrete Only!)

You'll only use the **discrete action variants**:
- ✅ **DiscreteBC** - Behavior Cloning baseline
- ✅ **DiscreteCQL** - Offline RL baseline
- ✅ **DiscreteDecisionTransformer** - Sequence modeling
- ✅ **DiscreteTACR** - Your novel contribution!

---

## ✅ What's Already Working

### Algorithms Implemented
- ✅ BC, DiscreteBC
- ✅ CQL, DiscreteCQL  
- ✅ DecisionTransformer, DiscreteDecisionTransformer
- ✅ **TACR, DiscreteTACR** (your contribution!)
- ✅ BCQ, IQL, AWAC (bonus options)

### Experiments Run Successfully
- ✅ **exp08:** DiscreteTACR on CartPole, Hopper, Atari Pong
  - With SLURM hyperparameter tuning!
  - Script: `run_discrete_TACR.py`
  - SLURM: `discrete_TACR_hyperparameter_tuning_two.sh`
- ✅ **exp07:** DiscreteDT on CartPole, Atari Pong
- ✅ **exp05:** CQL, BCQ on Hopper
- ✅ **exp01:** DT on Hopper-medium-expert
- ✅ **exp03, exp13:** FQE evaluation

### Infrastructure Present
- ✅ Dataset loaders: `get_cartpole()`, `get_minari()`, pickle loading
- ✅ FQE implementation for offline evaluation
- ✅ SLURM scripts working on cluster
- ✅ Environment evaluation utilities

---

## 🔴 What Needs to Be Done

### Priority 1: Standardization (Days 1-7)
**Problem:** Experiments scattered in notebooks, not reproducible

**Solution:**
1. **Unified training script** - based on `exp08/run_discrete_TACR.py`
2. **Config system** - YAML files for hyperparameters
3. **Pre-cluster tests** - prevent job failures
4. **Sepsis data wrapper** - integrate MIMIC into d3rlpy format

### Priority 2: Complete Algorithm Coverage (Days 8-14)
**Problem:** Not all algorithms tested on all benchmarks

**Solution:**
Complete the comparison matrix (discrete actions only):

| Algorithm | CartPole | Pong | Sepsis |
|-----------|----------|------|--------|
| DiscreteBC | ⬜ | ⬜ | ⬜ |
| DiscreteCQL | ⬜ | ⬜ | ⬜ |
| DiscreteDT | ⬜ | ⬜ | ⬜ |
| DiscreteTACR | ✅ | ✅ | ⬜ |

### Priority 3: Run & Evaluate (Days 15-25)
- Submit all experiments to cluster
- Collect results
- Run FQE for sepsis
- Analyze & compare

---

## 📅 Detailed 25-Day Timeline

### **Week 1: Infrastructure & Testing (Days 1-7)**

#### Day 1-2: Testing Infrastructure ⚠️ CRITICAL
**Goal:** Prevent cluster job failures

**Create:**
1. `tests/import_test.py`
```python
"""Quick import validation before cluster submission"""
def test_imports():
    import d3rlpy
    from d3rlpy.algos import BC, CQL
    from d3rlpy.algos.transformer import DecisionTransformer, TACR
    from d3rlpy.algos.transformer import DiscreteDecisionTransformer, DiscreteTACR
    print("✅ All imports OK")
```

2. `tests/training_step_test.py`
```python
"""Test one training step for each algorithm"""
def test_discrete_tacr_step():
    dataset, env = d3rlpy.datasets.get_cartpole()
    tacr = d3rlpy.algos.DiscreteTACRConfig(batch_size=64).create()
    # Run 1 step only
    tacr.fit(dataset, n_steps=10, n_steps_per_epoch=10)
    print("✅ DiscreteTACR training OK")
```

3. `tests/run_all_pre_cluster_tests.sh`
```bash
#!/bin/bash
echo "🔍 Pre-Cluster Validation"
python -m tests.import_test || exit 1
python -m tests.training_step_test || exit 1
echo "✅ Ready for cluster!"
```

**Deliverable:** Validation script that runs in <2 minutes

---

#### Day 3-5: Unified Training System
**Goal:** One script to train any algorithm on any dataset

**Approach:** Adapt your working `exp08/run_discrete_TACR.py`

**Create:** `training/train.py`
```python
#!/usr/bin/env python3
"""
Unified training script for all algorithms
Based on exp08/run_discrete_TACR.py
"""
import argparse
import d3rlpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['bc', 'cql', 'dt', 'tacr'], required=True)
    parser.add_argument('--dataset', choices=['cartpole', 'pong', 'hopper', 'sepsis'], required=True)
    parser.add_argument('--config', type=str, help='Path to YAML config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--debug', action='store_true', help='Fast test mode')
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset == 'cartpole':
        dataset, env = d3rlpy.datasets.get_cartpole()
        target_return = 200
    elif args.dataset == 'pong':
        dataset, env = d3rlpy.datasets.get_minari('atari/pong/expert-v0')
        target_return = 20
    elif args.dataset == 'sepsis':
        dataset = load_sepsis_dataset()  # New loader needed
        env = None  # No environment for sepsis
        target_return = 1.0
    
    # Create algorithm (ALL DISCRETE!)
    if args.algo == 'tacr':
        algo = d3rlpy.algos.DiscreteTACRConfig(
            batch_size=64,
            actor_learning_rate=1e-4,
            context_size=20,
            num_heads=8,
            num_layers=6,
            # ... load from config if provided
        ).create(device=args.device)
    elif args.algo == 'bc':
        algo = d3rlpy.algos.DiscreteBCConfig(
            batch_size=64,
            learning_rate=1e-4,
        ).create(device=args.device)
    elif args.algo == 'cql':
        algo = d3rlpy.algos.DiscreteCQLConfig(
            batch_size=64,
            learning_rate=1e-4,
        ).create(device=args.device)
    elif args.algo == 'dt':
        algo = d3rlpy.algos.DiscreteDecisionTransformerConfig(
            batch_size=64,
            learning_rate=1e-4,
            context_size=20,
        ).create(device=args.device)
    
    # Training
    n_steps = 10 if args.debug else 100000
    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=min(1000, n_steps),
        experiment_name=f"{args.algo}_{args.dataset}_{args.seed}",
    )

if __name__ == "__main__":
    main()
```

**Test locally:**
```bash
# Fast debug run
python training/train.py --algo tacr --dataset cartpole --debug

# Real run
python training/train.py --algo tacr --dataset cartpole
```

**Deliverable:** Working training script for all your benchmarks

---

#### Day 6-7: Sepsis Dataset Integration
**Goal:** Get sepsis data working with d3rlpy

**Create:** `datasets/sepsis_loader.py`
```python
"""
Load MIMIC sepsis data in d3rlpy format
Based on exp11_mimic_exploration
"""
import pandas as pd
import numpy as np
from d3rlpy.dataset import MDPDataset, Episode

def load_sepsis_dataset(reward_mode='terminal', data_path=None):
    """
    Load sepsis patient trajectories
    
    Args:
        reward_mode: 'terminal' or 'dense'
        data_path: Path to processed MIMIC data
    
    Returns:
        MDPDataset compatible with d3rlpy
    """
    # Load your MIMIC data
    df = pd.read_pickle(data_path or 'data/sepsis/processed.pkl')
    
    episodes = []
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id]
        
        # Extract observations (clinical features)
        observations = patient_data[FEATURE_COLUMNS].values
        
        # Discrete actions (fluids_bin * 5 + vasopressor_bin)
        actions = compute_discretized_actions(patient_data)
        
        # Rewards
        if reward_mode == 'terminal':
            rewards = compute_terminal_rewards(patient_data)
        else:
            rewards = compute_dense_rewards(patient_data)
        
        # Create episode
        episode = Episode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminated=bool(patient_data['mortality'].iloc[-1])
        )
        episodes.append(episode)
    
    return MDPDataset(episodes=episodes)
```

**Test:**
```python
# Test sepsis loading
dataset = load_sepsis_dataset(reward_mode='terminal')
print(f"Loaded {len(dataset.episodes)} patient trajectories")

# Test with DiscreteBC
bc = d3rlpy.algos.DiscreteBCConfig().create()
bc.fit(dataset, n_steps=100)  # Quick test
```

**Deliverable:** Sepsis data loadable like other datasets

---

### **Week 2: SLURM & Evaluation (Days 8-14)**

#### Day 8-10: Standardized SLURM Templates
**Goal:** Reliable cluster submission

**Create:** `slurm/train_template.slurm`
```bash
#!/bin/bash
#SBATCH --job-name={ALGO}_{DATASET}
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/{ALGO}_{DATASET}_%j.out
#SBATCH --error=logs/{ALGO}_{DATASET}_%j.err

# Environment setup (your existing setup from exp08)
export D3RLPY_DATASETS_PATH="/gpfs/data/.../d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/.../minari_data"

source ~/.bashrc
conda activate d3rlpy_env

# IMPORTANT: Run validation tests first!
echo "Running pre-cluster validation..."
bash tests/run_all_pre_cluster_tests.sh
if [ $? -ne 0 ]; then
    echo "❌ Validation failed! Job cancelled."
    exit 1
fi

# Run training
echo "Starting training: {ALGO} on {DATASET}"
python training/train.py \
    --algo {ALGO} \
    --dataset {DATASET} \
    --seed {SEED} \
    --device cuda:0
```

**Create:** `slurm/submit_experiments.py`
```python
#!/usr/bin/env python3
"""
Batch experiment submission
Usage: python slurm/submit_experiments.py --algorithms bc cql dt tacr --datasets cartpole hopper --seeds 1 2 3
"""
import argparse
import subprocess
import os

def submit_job(algo, dataset, seed):
    # Generate job script from template
    with open('slurm/train_template.slurm') as f:
        template = f.read()
    
    job_script = template.format(
        ALGO=algo,
        DATASET=dataset,
        SEED=seed
    )
    
    job_file = f'slurm/generated_jobs/{algo}_{dataset}_{seed}.slurm'
    os.makedirs('slurm/generated_jobs', exist_ok=True)
    
    with open(job_file, 'w') as f:
        f.write(job_script)
    
    # Submit
    result = subprocess.run(['sbatch', job_file], capture_output=True)
    print(f"Submitted: {algo} on {dataset} (seed {seed}) - {result.stdout.decode()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    for algo in args.algorithms:
        for dataset in args.datasets:
            for seed in args.seeds:
                if args.dry_run:
                    print(f"Would submit: {algo} on {dataset} seed {seed}")
                else:
                    submit_job(algo, dataset, seed)

if __name__ == "__main__":
    main()
```

**Test:**
```bash
# Dry run to preview jobs
python slurm/submit_experiments.py \
    --algorithms bc tacr \
    --datasets cartpole hopper \
    --seeds 1 2 3 \
    --dry-run

# Submit one test job
sbatch slurm/generated_jobs/bc_cartpole_1.slurm
```

**Deliverable:** One-command batch submission

---

#### Day 11-13: Evaluation Automation
**Goal:** Automated result collection

**Create:** `evaluation/evaluate_online.py`
```python
#!/usr/bin/env python3
"""
Online evaluation for benchmark environments
"""
import argparse
import d3rlpy
import numpy as np

def evaluate_policy(checkpoint_path, dataset_name, n_episodes=100, target_return=None):
    # Load policy
    if 'tacr' in checkpoint_path:
        algo = d3rlpy.algos.DiscreteTACR.from_file(checkpoint_path)
    elif 'dt' in checkpoint_path:
        algo = d3rlpy.algos.DiscreteDecisionTransformer.from_file(checkpoint_path)
    # ... etc
    
    # Load environment
    if dataset_name == 'cartpole':
        env = gym.make('CartPole-v1')
    elif dataset_name == 'hopper':
        env = gym.make('Hopper-v4')
    # ...
    
    # For DT/TACR: wrap with RTG
    if hasattr(algo, 'eval_target_return'):
        from d3rlpy.algos.transformer.base import DiscountedRTGWrapper
        algo = DiscountedRTGWrapper(algo, target_return=target_return)
    
    # Run episodes
    returns = []
    for i in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = algo.predict([obs])[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        returns.append(total_reward)
    
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns),
    }
```

**Create:** `evaluation/evaluate_fqe.py`
```python
#!/usr/bin/env python3
"""
FQE evaluation for sepsis (offline only)
"""
import argparse
import d3rlpy

def evaluate_with_fqe(policy_checkpoint, dataset, n_steps=10000):
    # Load trained policy
    policy = d3rlpy.load_learnable(policy_checkpoint)
    
    # Train FQE
    fqe = d3rlpy.ope.FQEConfig(
        learning_rate=1e-4,
    ).create()
    
    fqe.fit(
        dataset,
        n_steps=n_steps,
        target_algo=policy,
    )
    
    # Estimate value
    estimated_value = fqe.predict_value(
        dataset.observations,
        dataset.actions,
    )
    
    return estimated_value.mean()
```

**Deliverable:** Automated evaluation scripts

---

#### Day 14: Buffer & Documentation
- Fix any issues from test jobs
- Document workflow in README
- Prepare final experiment configs

---

### **Week 3: Run Experiments (Days 15-21)**

#### Day 15-16: BC & CQL Baselines
```bash
# Submit all BC experiments (discrete only!)
python slurm/submit_experiments.py \
    --algorithms bc \
    --datasets cartpole pong sepsis \
    --seeds 1 2 3

# Submit all CQL experiments
python slurm/submit_experiments.py \
    --algorithms cql \
    --datasets cartpole pong sepsis \
    --seeds 1 2 3
```

**Monitor:** `squeue -u $USER`

---

#### Day 17-18: DT Experiments
```bash
python slurm/submit_experiments.py \
    --algorithms dt \
    --datasets cartpole pong sepsis \
    --seeds 1 2 3
```

---

#### Day 19-20: TACR Experiments
```bash
python slurm/submit_experiments.py \
    --algorithms tacr \
    --datasets cartpole pong sepsis \
    --seeds 1 2 3
```

---

#### Day 21: Evaluation
```bash
# Online evaluation (CartPole, Pong only)
for checkpoint in models/{cartpole,pong}/*/checkpoint_*.pt; do
    python evaluation/evaluate_online.py --checkpoint $checkpoint
done

# FQE evaluation (Sepsis only - no environment available)
for checkpoint in models/sepsis/*/checkpoint_*.pt; do
    python evaluation/evaluate_fqe.py --checkpoint $checkpoint
done
```

---

### **Week 4: Analysis & Writing (Days 22-25)**

#### Day 22-23: Result Analysis
```python
# evaluation/analyze_results.py
import pandas as pd
import numpy as np

# Aggregate across seeds
results = []
for algo in ['bc', 'cql', 'dt', 'tacr']:
    for dataset in ['cartpole', 'pong', 'hopper', 'sepsis']:
        for seed in [1, 2, 3]:
            result = load_result(algo, dataset, seed)
            results.append(result)

df = pd.DataFrame(results)
summary = df.groupby(['algo', 'dataset']).agg(['mean', 'std'])
print(summary)
```

**Create comparison table:**

| Algorithm | CartPole | Pong | Sepsis (FQE) |
|-----------|----------|------|--------------|
| DiscreteBC | 100±5 | 10±2 | 0.3±0.1 |
| DiscreteCQL | 150±10 | 15±3 | 0.4±0.08 |
| DiscreteDT | 180±8 | 18±2 | 0.5±0.09 |
| DiscreteTACR | **195±5** | **19±1** | **0.6±0.07** |

**Hypothesis:** TACR should outperform baselines across all three discrete action domains.

---

#### Day 24: Visualization
- Learning curves
- Performance comparisons
- Statistical significance tests

---

#### Day 25: Buffer Day
- Handle failed jobs
- Rerun if needed
- Final documentation

---

## 🎯 Quick Start Guide

### Step 1: Run Pre-Flight Check
```bash
cd /path/to/d3rlpy
bash tests/run_all_pre_cluster_tests.sh
```

### Step 2: Test Training Locally
```bash
# Debug mode (fast test)
python training/train.py --algo tacr --dataset cartpole --debug

# Full local run (if you have time)
python training/train.py --algo bc --dataset cartpole
```

### Step 3: Submit to Cluster
```bash
# Submit one test job first
python slurm/submit_experiments.py \
    --algorithms bc \
    --datasets cartpole \
    --seeds 1 \
    --dry-run  # Preview first!

# If looks good, submit for real
python slurm/submit_experiments.py \
    --algorithms bc \
    --datasets cartpole \
    --seeds 1
```

### Step 4: Monitor & Scale Up
```bash
# Check job status
squeue -u $USER

# If successful, submit all experiments
python slurm/submit_experiments.py \
    --algorithms bc cql dt tacr \
    --datasets cartpole pong hopper sepsis \
    --seeds 1 2 3
```

---

## 📊 Experiment Matrix (Simplified - Discrete Only!)

**Total experiments:** 4 algorithms × 3 datasets × 3 seeds = **36 runs**

| Dataset | DiscreteBC | DiscreteCQL | DiscreteDT | DiscreteTACR | Total |
|---------|------------|-------------|------------|--------------|-------|
| CartPole | 3 | 3 | 3 | 3 | 12 |
| Pong | 3 | 3 | 3 | 3 | 12 |
| Sepsis | 3 | 3 | 3 | 3 | 12 |
| **Total** | **9** | **9** | **9** | **9** | **36** |

**Benefits of removing Hopper:**
- ✅ No continuous/discrete algorithm switching
- ✅ 25% fewer experiments (48 → 36)
- ✅ Simpler codebase (only discrete variants)
- ✅ Faster development time

**Estimated cluster time per run:** 2-6 hours  
**Total GPU hours:** 108-216 hours (25% reduction!)  
**Wall clock time (parallel):** 2-6 hours if enough GPUs

---

## 💡 Key Success Factors

### ✅ You Already Have
1. **Working TACR implementation** - your contribution!
2. **SLURM environment** - exp08 proves it works
3. **Dataset loaders** - get_cartpole(), get_minari()
4. **FQE framework** - ready for sepsis
5. **Reproducible notebooks** - know what works

### ⚠️ Critical Path
1. **Testing** (Days 1-2) - prevents wasted GPU time
2. **Training script** (Days 3-5) - enables systematic experiments
3. **Sepsis integration** (Days 6-7) - your novel contribution
4. **SLURM templates** (Days 8-10) - scales to all experiments

### 🎯 Timeline Confidence
**25 days:** ✅ **Achievable**  
**20 days:** ⚠️ Tight but possible  
**15 days:** ❌ Too aggressive

**Recommended buffer:** Keep 5 days for:
- Cluster queue times
- Failed jobs
- Dataset issues
- Result analysis

---

## 🚀 Ready to Start?

### Immediate Next Actions (Choose Your Path):

**Option A: Conservative (Recommended)**
1. Create testing infrastructure (Days 1-2)
2. Build on your exp08 success
3. Add sepsis integration
4. Scale systematically

**Option B: Aggressive**
1. Copy exp08 script → adapt for all algorithms
2. Wrap sepsis data
3. Submit test jobs immediately
4. Add tests while jobs run

**Option C: Targeted**
1. Focus on sepsis first (your novel contribution)
2. Use existing exp08 for benchmarks
3. Combine later for thesis

### My Recommendation: **Option A**

Your exp08 proves the infrastructure works. Let's build on that foundation systematically.

**Shall we start by creating the testing infrastructure?**
