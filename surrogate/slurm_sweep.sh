#!/bin/bash
#SBATCH --job-name=sweep-rsnn
#SBATCH --partition=dev_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=00:05:00
#SBATCH --output=logs_slurm/sweep_%j.out
#SBATCH --error=logs_slurm/sweep_%j.err

# Load Python module and activate environment if you have one
module load devel/miniforge

# Ensure logging directories exist
mkdir -p logs_slurm
mkdir -p /scratch/$USER/wandb_logs

# Install dependencies (optional if env already has them)
pip install -r requirements.txt

# Run the sweep with unbuffered output for real-time logging
python -u 3_hyperparameter_search_rsnn.py \
    --sweep-config sweep_hyperparameter_rsnn.yaml \
    --project SpikeSynth-Surrogate-Sweep \
    --logging-directory /scratch/$USER/wandb_logs

