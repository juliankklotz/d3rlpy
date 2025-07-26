import os

os.environ["D3RLPY_DATASETS_PATH"] = "/home/julian/programming/data/d3rlpy_data_test"
os.makedirs(os.environ["D3RLPY_DATASETS_PATH"], exist_ok=True)
os.environ["MINARI_DATASETS_PATH"] = "/home/julian/programming/data/d3rlpy_data_test/minari_data"
os.makedirs(os.environ["MINARI_DATASETS_PATH"], exist_ok=True)

import d3rlpy

dataset, env = d3rlpy.datasets.get_cartpole()

seed = 1

dataset_name = "cartpole"
# fix seed
d3rlpy.seed(seed)
d3rlpy.envs.seed_env(env, seed)

if "cartpole" in dataset_name:
    target_return = 200
else:
    raise ValueError("unsupported dataset")

discrete_tacr = d3rlpy.algos.DiscreteTACRConfig(
    batch_size=64,
    actor_learning_rate=1e-4,
    actor_optim_factory=d3rlpy.optimizers.AdamWFactory(
        weight_decay=1e-4,
        clip_grad_norm=0.25,
        lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
            warmup_steps=100#10000
        ),
    ),
    actor_encoder_factory=d3rlpy.models.VectorEncoderFactory(
        [128],
        exclude_last_activation=True,
    ),
    observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
    position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
    context_size=20,
    num_heads=1,
    num_layers=3,
    max_timestep=1000,
    compile_graph=True,
).create(device="cpu")

discrete_tacr.fit(
    dataset,
    n_steps=100,# 100000,
    n_steps_per_epoch=10,# 1000,
    save_interval=10,
    eval_env=env,
    eval_target_return=target_return,
    experiment_name=f"Discrete_TACR_{dataset_name}_{seed}",
)