#!/bin/bash
#SBATCH --array=0-8
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --mem=4G
#SBATCH --time=06:00:00

# Datasets list
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
jobname="dt_${dataset}"

# Logs
logdir="slurm_logs"
mkdir -p "$logdir"
stdout_file="${logdir}/${jobname}.out"
stderr_file="${logdir}/${jobname}.err"

# Redirect stdout and stderr
exec > >(tee -a "$stdout_file")
exec 2> >(tee -a "$stderr_file" >&2)

# Print task info
echo "Starting job: $jobname"
echo "Running on node: $(hostname)"
echo "Time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"


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
  --n_epochs 100
