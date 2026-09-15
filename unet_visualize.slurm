#!/bin/bash
#SBATCH --job-name=unet_inference
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=unet_inference_%j.out
#SBATCH --error=unet_inference_%j.err

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/bin/activate"
conda activate torch

python -u src/inference_unet.py \
    --data_root /home/groups/comp3710/OASIS \
    --checkpoint checkpoints/best_unet.pth \
    --output_dir outputs