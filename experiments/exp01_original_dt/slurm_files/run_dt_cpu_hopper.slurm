#!/bin/bash
#SBATCH -J dt_cpu
#SBATCH --partition=zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH -t 06:00:00
#SBATCH --mem 4G

module load miniconda3/24.1.2

# Properly initialize Conda
source /opt/sw/conda/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda deactivate
conda activate d3rlpy_dev_requirements_py310

echo "Python executable: $(which python)"
echo "Conda environment: $(conda env list | grep '*' | awk '{print $1}')"

python /gpfs/data/fs72297/jklotz/programming/cloned_repos/forked_repos_for_master_thesis/d3rlpy/experiments/exp01_original_dt/run_decision_transformer.py \
  --dataset hopper-medium-expert-v2 \
  --n_steps 10000 \
  --n_epochs 100
