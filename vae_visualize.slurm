#!/bin/bash
#SBATCH --job-name=vae_visualize
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=vae_visualize_%j.out
#SBATCH --error=vae_visualize_%j.err

cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

source "$HOME/miniconda3/bin/activate"
conda activate torch

python -u src/visualize_vae.py \
    --data_root /home/groups/comp3710/OASIS \
    --checkpoint checkpoints/best_vae.pth \
    --output_dir outputs

echo "Finished: $(date)"