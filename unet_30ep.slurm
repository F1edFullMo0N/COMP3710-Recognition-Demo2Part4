#!/bin/bash
#SBATCH --job-name=unet_30ep
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=unet_30ep_%j.out
#SBATCH --error=unet_30ep_%j.err

cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Python: $(which python)"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

nvidia-smi

echo "Starting U-Net training..."

python -u src/train_unet.py \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --workers 4 \
    --data_root /home/groups/comp3710/OASIS

echo "Finished: $(date)"