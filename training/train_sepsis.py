#!/usr/bin/env python3
"""Train one discrete offline RL algo on the Sepsis/MIMIC-IV dataset, then run FQE.

No simulator available — evaluation is offline-only via FQE, on a held-out
test fold (5-fold stratified CV by survival outcome).
BC/CQL use DiscreteFQE; DT/TACR use DiscreteFQETrajectory (sequence-aware).

Usage:
    python training/train_sepsis.py --algo discrete_bc --seed 0 --fold 0
    python training/train_sepsis.py --algo discrete_tacr --seed 1 --fold 0 --device cuda:0
    SEPSIS_DATA_DIR=/data/sepsis python training/train_sepsis.py --algo discrete_cql --seed 2 --fold 0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for fqe_policy_value

import d3rlpy
import d3rlpy.preprocessing
import d3rlpy.models
from d3rlpy.logging import UnifiedFileAdapterFactory
from d3rlpy.ope import DiscreteFQE, DiscreteFQETrajectory, FQEConfig
from d3rlpy.sepsis_loader import get_sepsis_splits, get_sepsis_dev_buffer

# Sepsis dataset constants
N_ACTIONS = 25     # 5 fluid levels × 5 vasopressor levels
OBS_DIM = 47       # see sepsis_loader.OBSERVATION_COLUMNS
CONTEXT_SIZE = 20  # median ICU stay length — MUST match hardcoded FQE trajectory
                    # length=20 in QLearningAlgoBase.fitter() (batch_type="trajectory").
                    # Sequence-length ablations (K != 20) cannot be FQE-evaluated as-is.

HPARAMS = {
    "discrete_bc": dict(batch_size=64, learning_rate=1e-3),
    "discrete_cql": dict(batch_size=64),
    "discrete_dt": dict(
        batch_size=64, context_size=CONTEXT_SIZE, num_heads=4, num_layers=6,
        max_timestep=100,
    ),
    "discrete_tacr": dict(
        batch_size=64, context_size=CONTEXT_SIZE, num_heads=4, num_layers=6,
        actor_learning_rate=1e-4, max_timestep=100,
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        compile_graph=False,
    ),
}

N_STEPS = 100_000
N_STEPS_PER_EPOCH = 1_000
FQE_N_STEPS = 50_000
N_FOLDS = 5

# RTG used to query DT/TACR at FQE bootstrap time, per reward_mode.
# terminal reward: RTG is unambiguous (+1 = desired survival).
# mixed reward: no natural RTG scale: use primary target from thesis 5.2.3.
EVAL_RTG_BY_REWARD_MODE = {
    "terminal": 1.0,
    "mixed": 15.0,
    "dense": None,  # dense reward has no principled fixed target; use data-derived RTG
}


def build_algo(algo_name: str, device: str, hp_override: Optional[dict] = None) -> d3rlpy.base.LearnableBase:
    # base config + tuned/searched overlay (hp_override wins on key collisions)
    hp = dict(HPARAMS[algo_name])
    if hp_override:
        hp.update(hp_override)
    obs_scaler = d3rlpy.preprocessing.StandardObservationScaler()

    if algo_name == "discrete_bc":
        return d3rlpy.algos.DiscreteBCConfig(
            observation_scaler=obs_scaler, **hp
        ).create(device=device)

    if algo_name == "discrete_cql":
        return d3rlpy.algos.DiscreteCQLConfig(
            observation_scaler=obs_scaler, **hp
        ).create(device=device)

    if algo_name == "discrete_dt":
        return d3rlpy.algos.DiscreteDecisionTransformerConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            **hp,
        ).create(device=device)

    if algo_name == "discrete_tacr":
        return d3rlpy.algos.DiscreteTACRConfig(
            observation_scaler=obs_scaler,
            reward_scaler=d3rlpy.preprocessing.StandardRewardScaler(),
            actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
                [256, 256], exclude_last_activation=True
            ),
            actor_optim_factory=d3rlpy.optimizers.AdamWFactory(
                weight_decay=1e-4,
                clip_grad_norm=0.25,
                lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                    warmup_steps=1_000
                ),
            ),
            **hp,
        ).create(device=device)

    raise ValueError(f"Unknown algo: {algo_name}")


def run_fqe(
    algo: d3rlpy.base.LearnableBase,
    fqe_fit_dataset: d3rlpy.dataset.ReplayBuffer,
    eval_dataset: d3rlpy.dataset.ReplayBuffer,
    algo_name: str,
    reward_mode: str,
    device: str,
    experiment_name: str,
    fqe_n_steps: int = FQE_N_STEPS,
) -> float:
    """Fit FQE on `fqe_fit_dataset`, report the policy value on `eval_dataset`.

    The reported metric is the thesis section 5.5.2 objective (see
    training/fqe_policy_value.py): the FQE policy value averaged over the
    held-out trajectories, evaluated at each trajectory's INITIAL state under
    the POLICY's action distribution — NOT the earlier per-transition average
    over the logged clinician actions (which measured the critic's opinion of
    the clinician, not the value of pi).
    """
    import numpy as np
    from fqe_policy_value import fqe_policy_value_per_trajectory

    is_transformer = algo_name in ("discrete_dt", "discrete_tacr")
    target_rtg = EVAL_RTG_BY_REWARD_MODE[reward_mode] if is_transformer else None

    fqe_config = FQEConfig(batch_size=64, n_critics=1, target_return=target_rtg)

    if is_transformer:
        fqe = DiscreteFQETrajectory(algo, fqe_config, device=device)
        fit_kwargs = dict(batch_type="trajectory")
    else:
        fqe = DiscreteFQE(algo, fqe_config, device=device)
        fit_kwargs = dict()

    fqe.fit(
        fqe_fit_dataset,
        n_steps=fqe_n_steps,
        n_steps_per_epoch=min(1_000, fqe_n_steps),
        experiment_name=f"fqe_{experiment_name}",
        logger_adapter=UnifiedFileAdapterFactory(),
        show_progress=False,
        **fit_kwargs,
    )

    # Correct policy-value metric: per-trajectory V_hat(pi) at initial states.
    v = fqe_policy_value_per_trajectory(
        fqe, algo, eval_dataset, algo_name, target_return=target_rtg
    )
    if len(v) == 0:
        return float("nan")
    v_hat = float(np.mean(v))
    print(
        f"FQE policy value V_hat={v_hat:.4f} +/- {float(np.std(v)):.4f} "
        f"(n={len(v)} trajectories)"
    )
    # persist per-trajectory values for later aggregation (mean/CI, paired tests)
    try:
        import os
        os.makedirs("fqe_values", exist_ok=True)
        np.save(f"fqe_values/{experiment_name}_vpertraj.npy", v)
    except Exception as e:
        print(f"  (could not save per-trajectory values: {e})")
    return v_hat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        choices=["discrete_bc", "discrete_cql", "discrete_dt", "discrete_tacr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="final", choices=["tune", "final"],
                        help="tune: fit on dev-train, eval on VAL (for HP selection). "
                             "final: fit on full DEV, eval ONCE on the locked TEST set.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_dir", default=None,
                        help="Path to MIMIC-IV sepsis CSV (default: $SEPSIS_DATA_DIR)")
    parser.add_argument("--reward_mode", default="terminal",
                        choices=["terminal", "dense", "mixed"],
                        help="terminal: ±1 survival/death; dense: -SOFA each step; mixed: 0.5*SOFA-delta + ±15 terminal")
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fraction of stays locked as the final test set.")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of the DEV set used as validation (tune mode).")
    parser.add_argument("--split_seed", type=int, default=0,
                        help="Seed fixing the dev/test lock (same across all algos/seeds).")
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--hp_json", default=None,
                        help="Path to tuned_configs.json; in final mode, loads the "
                             "tuned HP for this algo+reward_mode. Ignored in tune mode.")
    parser.add_argument("--hp_override", default=None,
                        help="JSON string of HP overrides (used by the tuning script "
                             "to inject a grid point). Wins over --hp_json.")
    parser.add_argument("--skip_fqe", action="store_true",
                        help="Skip FQE eval (faster, use for debugging)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: tiny n_steps for train + FQE, to verify "
                             "the full pipeline runs end-to-end without burning "
                             "cluster time. Does NOT produce usable results.")
    args = parser.parse_args()

    train_n_steps = args.n_steps
    fqe_n_steps = FQE_N_STEPS
    steps_per_epoch = N_STEPS_PER_EPOCH
    if args.smoke:
        train_n_steps = 5
        fqe_n_steps = 5
        steps_per_epoch = 5
        print(">>> SMOKE TEST MODE: 5 train steps, 5 FQE steps (results not usable)")

    d3rlpy.seed(args.seed)

    data_dir = args.data_dir or os.environ.get("SEPSIS_DATA_DIR")
    if not data_dir:
        print("ERROR: set --data_dir or SEPSIS_DATA_DIR", file=sys.stderr)
        sys.exit(1)

    # Leakage-free 3-way split. The dev/test lock is fixed by --split_seed so the
    # locked test set is IDENTICAL across every algorithm, reward, and run seed.
    if args.mode == "tune":
        print(f"[TUNE] dev-train fit, VAL eval ({args.reward_mode}) ...")
        fit_dataset, eval_dataset, _locked_test = get_sepsis_splits(
            data_dir=data_dir, test_frac=args.test_frac, holdout=True,
            val_frac=args.val_frac, seed=args.split_seed, reward_mode=args.reward_mode,
        )
        eval_name = "val"
    else:  # final
        print(f"[FINAL] full-dev fit, LOCKED-TEST eval ({args.reward_mode}) ...")
        fit_dataset, eval_dataset = get_sepsis_dev_buffer(
            data_dir=data_dir, test_frac=args.test_frac, seed=args.split_seed,
            reward_mode=args.reward_mode,
        )
        eval_name = "test"
    print(f"  fit:  episodes={len(fit_dataset.episodes)} transitions={fit_dataset.transition_count}")
    print(f"  {eval_name}: episodes={len(eval_dataset.episodes)} transitions={eval_dataset.transition_count}")

    # Resolve tuned/searched hyperparameters.
    hp_override = None
    if args.hp_override:
        hp_override = json.loads(args.hp_override)
        print(f"HP override (grid point): {hp_override}")
    elif args.hp_json and args.mode == "final":
        with open(args.hp_json) as f:
            tuned = json.load(f)
        hp_override = tuned.get(args.algo, {}).get(args.reward_mode)
        if hp_override is None:
            print(f"WARNING: no tuned HP for {args.algo}/{args.reward_mode} in {args.hp_json}; using defaults")
        else:
            print(f"Tuned HP for {args.algo}/{args.reward_mode}: {hp_override}")

    algo = build_algo(args.algo, args.device, hp_override=hp_override)
    experiment_name = f"{args.algo}_sepsis_{args.mode}_{args.reward_mode}_seed{args.seed}"

    print(f"algo={args.algo}  mode={args.mode}  seed={args.seed}  n_steps={train_n_steps}")
    print(f"python={sys.executable}  device={args.device}")

    algo.fit(
        fit_dataset,
        n_steps=train_n_steps,
        n_steps_per_epoch=steps_per_epoch,
        save_interval=steps_per_epoch * 10,
        experiment_name=experiment_name,
        logger_adapter=UnifiedFileAdapterFactory(),
        show_progress=False,
    )

    model_path = f"{experiment_name}_final.pt"
    algo.save(model_path)
    print(f"Model saved: {model_path}")

    if not args.skip_fqe:
        print(f"Running FQE (fit on {args.mode}-fit set, eval on {eval_name}) ...")
        run_fqe(algo, fit_dataset, eval_dataset, args.algo, args.reward_mode,
                args.device, experiment_name, fqe_n_steps=fqe_n_steps)

    if args.smoke:
        print(">>> SMOKE TEST PASSED: full pipeline ran end-to-end")


if __name__ == "__main__":
    main()
