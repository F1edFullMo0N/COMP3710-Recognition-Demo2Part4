#!/bin/bash
#SBATCH --job-name=vae_20ep
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=vae_20ep_%j.out
#SBATCH --error=vae_20ep_%j.err

cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Python: $(which python)"

python -c "
import torch
import numpy
import PIL
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('NumPy:', numpy.__version__)
print('Pillow:', PIL.__version__)
"

nvidia-smi

echo "Starting 20-epoch VAE training..."

python -u src/train_vae.py \
    --epochs 20 \
    --batch_size 16 \
    --latent_dim 2 \
    --data_root /home/groups/comp3710/OASIS

echo "Finished: $(date)"