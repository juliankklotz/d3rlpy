"""Example: Training DiscreteBC on MIMIC-IV Sepsis Dataset

This example demonstrates how to use the sepsis dataset loader
with discrete action space for offline RL experiments.
"""

import d3rlpy
from d3rlpy.constants import ActionSpace

# Set data directory (optional - defaults to ../data or uses SEPSIS_DATA_DIR env var)
# import os
# os.environ["SEPSIS_DATA_DIR"] = "/path/to/data"

# Load sepsis dataset with discrete actions (25 bins: 5x5 fluid×vasopressor grid)
print("Loading sepsis dataset...")
dataset = d3rlpy.datasets.get_sepsis(action_space=ActionSpace.DISCRETE)

print(f"Dataset loaded:")
print(f"  - Episodes: {len(dataset.episodes)}")
print(f"  - Transitions: {dataset.transition_count}")
print(f"  - Action space: {dataset.dataset_info.action_space}")
print(f"  - Action size: {dataset.dataset_info.action_size}")
print(f"  - Observation shape: {dataset.dataset_info.observation_shape}")

# Sample a transition to inspect
sample = dataset.sample_transition()
print(f"\nSample transition:")
print(f"  - Observation shape: {sample.observation.shape}")
print(f"  - Action: {sample.action}")
print(f"  - Reward (SOFA score): {sample.reward}")

# Train DiscreteBC algorithm
print("\n" + "="*60)
print("Training DiscreteBC...")
print("="*60)

bc = d3rlpy.algos.DiscreteBCConfig(
    batch_size=256,
    learning_rate=1e-4,
    encoder_factory=d3rlpy.models.VectorEncoderFactory([256, 256]),
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
).create(device="cpu")  # Change to "cuda:0" for GPU

bc.fit(
    dataset,
    n_steps=10000,
    n_steps_per_epoch=1000,
    save_interval=10,
    experiment_name="DiscreteBC_Sepsis",
)

print("\n✓ Training complete!")
print(f"Model saved to: d3rlpy_logs/DiscreteBC_Sepsis_*/")

# Evaluate with FQE (Fitted Q Evaluation)
print("\n" + "="*60)
print("Running FQE evaluation...")
print("="*60)

fqe = d3rlpy.ope.DiscreteFQEConfig(
    learning_rate=1e-4,
    encoder_factory=d3rlpy.models.VectorEncoderFactory([256, 256]),
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
).create(device="cpu")

fqe.fit(
    dataset,
    n_steps=5000,
    n_steps_per_epoch=1000,
    evaluate_scorer=d3rlpy.metrics.evaluate_qlearning_with_environment,  # Requires env
    save_interval=10,
    experiment_name="FQE_DiscreteBC_Sepsis",
)

print("\n✓ Evaluation complete!")
