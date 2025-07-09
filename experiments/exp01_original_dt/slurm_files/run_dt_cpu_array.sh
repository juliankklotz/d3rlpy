#!/bin/bash
#SBATCH --array=0-8
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --job-name=dt_job_%a
#SBATCH --output=slurm_logs/cpu_array/dt_job_%a.out
#SBATCH --error=slurm_logs/cpu_array/dt_job_%a.err

# Create logs directory if it doesn't exist
mkdir -p slurm_logs
mkdir -p slurm_logs/cpu_array
# Dataset list
datasets=(
  "halfcheetah-medium-v2"
  "halfcheetah-medium-expert-v2"
  "halfcheetah-medium-replay-v2"
  "hopper-medium-v2"
  "hopper-medium-expert-v2"
  "hopper-medium-replay-v2"
  "walker2d-medium-v2"
  "walker2d-medium-replay-v2"
  "walker2d-medium-expert-v2"
)

dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
module load miniconda3/24.1.2

# Properly initialize Conda
source /opt/sw/conda/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda deactivate
conda activate d3rlpy_dev_requirements_py310

echo "Python executable: $(which python)"
echo "Conda environment: $(conda env list | grep '*' | awk '{print $1}')"
# Debug print
echo "Running Decision Transformer on dataset: $dataset"

python /gpfs/data/fs72297/jklotz/programming/cloned_repos/forked_repos_for_master_thesis/d3rlpy/experiments/exp01_original_dt/run_decision_transformer.py \
  --dataset $dataset \
  --n_steps 10000 \
  --n_epochs 100 \
  --prefix_experiment_name "cpu_array/" \
  --n_trials 1 \
  --patience 2 
