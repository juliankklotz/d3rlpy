#!/usr/bin/env python3
"""
train_discrete_tacr.py

Training script for Discrete TACR algorithm using d3rlpy.
Parses command-line arguments to configure and run the training.
"""
import argparse
import gymnasium as gym
from d3rlpy.algos import DiscreteTACRConfig
from d3rlpy.datasets import MDPDataset
from d3rlpy.logging import UnifiedFileAdapterFactory


def main():
    parser = argparse.ArgumentParser(description="Train Discrete TACR with d3rlpy")
    parser.add_argument('--algo', type=str, default='discrete_tacr',
                        help='Algorithm to run (must be "discrete_tacr")')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the MDPDataset HDF5 file')
    parser.add_argument('--env', type=str, default=None,
                        help='Gym environment ID for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--actor_learning_rate', type=float, required=True,
                        help='Learning rate for the actor')
    parser.add_argument('--critic_learning_rate', type=float, required=True,
                        help='Learning rate for the critic')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Weight of the Q-value term in the actor loss')
    parser.add_argument('--tau', type=float, required=True,
                        help='Soft-update coefficient for target networks')
    parser.add_argument('--n_critics', type=int, default=2,
                        help='Number of critics in the ensemble')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in the Transformer')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of Transformer layers')
    parser.add_argument('--context_size', type=int, default=20,
                        help='Sequence context length')
    parser.add_argument('--max_timestep', type=int, default=1000,
                        help='Maximum timestep for positional encoding')
    parser.add_argument('--compile_graph', action='store_true',
                        help='Enable JIT compilation and CUDAGraph')
    parser.add_argument('--n_steps', type=int, default=1000000,
                        help='Total number of training steps')
    parser.add_argument('--n_steps_per_epoch', type=int, default=1000,
                        help='Number of steps per epoch')
    parser.add_argument('--save_interval', type=int, default=50000,
                        help='Steps between checkpoint saves')
    parser.add_argument('--eval_interval', type=int, default=50000,
                        help='Steps between evaluations')
    parser.add_argument('--eval_target_return', type=float, default=0.0,
                        help='Target return for evaluation stopping')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Directory name for logging and saving checkpoints')
    args = parser.parse_args()

    if args.algo != 'discrete_tacr':
        raise ValueError(f"Algorithm {args.algo} not supported. Use 'discrete_tacr'.")

    # Load offline dataset
    dataset = MDPDataset(args.dataset)

    # Build evaluation environment
    eval_env = gym.make(args.env) if args.env else None

    # Create DiscreteTACR algorithm
    algo = DiscreteTACRConfig(
        batch_size=args.batch_size,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha=args.alpha,
        tau=args.tau,
        n_critics=args.n_critics,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        context_size=args.context_size,
        max_timestep=args.max_timestep,
        compile_graph=args.compile_graph,
    ).create(device=args.device)

    # Start training
    algo.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        save_interval=args.save_interval,
        eval_env=eval_env,
        eval_interval=args.eval_interval,
        eval_target_return=args.eval_target_return,
        experiment_name=args.logdir,
        logger_adapter=UnifiedFileAdapterFactory(),
        n_trials=50,
        eval_gaps=1
    )

if __name__ == '__main__':
    main()
