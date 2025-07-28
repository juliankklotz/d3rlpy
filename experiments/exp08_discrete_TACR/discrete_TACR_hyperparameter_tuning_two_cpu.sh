#!/bin/bash
#SBATCH --job-name=disc_tacr_tune
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --array=0-1


# Hyperparameter grids
ACTOR_LRS=(0.0001 0.0003)
CRITIC_LRS=(0.0001 0.0003)
ALPHAS=(1.0 2.5)
TAUS=(0.005 0.01)
CONTEXT_SIZES=(20 50)

# Grid sizes
NA=${#ACTOR_LRS[@]}      # 2
NC=${#CRITIC_LRS[@]}     # 2
NA2=${#ALPHAS[@]}        # 2
NT=${#TAUS[@]}           # 2
NS=${#CONTEXT_SIZES[@]}  # 2

# Linear index
IDX=$SLURM_ARRAY_TASK_ID  # 0..31

# Decode indices (actor lr slowest, context fastest)
i_actor=$(( IDX / (NC * NA2 * NT * NS) ))
rem1=$(( IDX % (NC * NA2 * NT * NS) ))
i_critic=$(( rem1 / (NA2 * NT * NS) ))
rem2=$(( rem1 % (NA2 * NT * NS) ))
i_alpha=$(( rem2 / (NT * NS) ))
rem3=$(( rem2 % (NT * NS) ))
i_tau=$(( rem3 / NS ))
i_context=$(( rem3 % NS ))

# Select values
ACTOR_LR=${ACTOR_LRS[$i_actor]}
CRITIC_LR=${CRITIC_LRS[$i_critic]}
ALPHA=${ALPHAS[$i_alpha]}
TAU=${TAUS[$i_tau]}
CONTEXT=${CONTEXT_SIZES[$i_context]}

echo "Job $IDX → actor_lr=$ACTOR_LR, critic_lr=$CRITIC_LR, alpha=$ALPHA, tau=$TAU, context_size=$CONTEXT"
export D3RLPY_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data"
export MINARI_DATASETS_PATH="/gpfs/data/fs72297/jklotz/programming_data/d3rlpy_data/minari_data"

# Launch training
/gpfs/data/fs72297/jklotz/.conda/envs/d3rlpy_dev_requirements_py310/bin/python /gpfs/data/fs72297/jklotz/programming/cloned_repos/forked_repos_for_master_thesis/d3rlpy/experiments/exp08_discrete_TACR/run_discrete_tacr.py \
  --algo discrete_tacr \
  --dataset cartpole \
  --env CartPole-v1 \
  --device cpu \
  --batch_size 128 \
  --actor_learning_rate $ACTOR_LR \
  --critic_learning_rate $CRITIC_LR \
  --alpha $ALPHA \
  --tau $TAU \
  --n_critics 4 \
  --num_heads 8 \
  --num_layers 6 \
  --context_size $CONTEXT \
  --max_timestep 200 \
  --compile_graph False \
  --n_steps 200000 \
  --n_steps_per_epoch 1000 \
  --save_interval 10000 \
  --eval_interval 10000 \
  --eval_target_return 200 \
  --logdir logs/tune_ar${ACTOR_LR}_cr${CRITIC_LR}_α${ALPHA}_τ${TAU}_ctx${CONTEXT}
