#!/bin/bash
#SBATCH --job-name=nano4m_v5_final
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/home/logs/nano4m_v5_%j.out
#SBATCH --error=/path/to/home/logs/nano4m_v5_%j.err

set -euo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi | head -16 || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

NANO4M_DIR=/path/to/nanofm
CONFIG=$NANO4M_DIR/cfgs/nano4M/animal_full_5mod_v5.yaml

# wandb: env var bypasses the CLI 40-char length check
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_DIR=$NANO4M_DIR/wandb

# Use our scratch copy of nano4M (has the completed fourm.py)
export PYTHONPATH=$NANO4M_DIR:${PYTHONPATH:-}

cd "$NANO4M_DIR"
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_training.py --config "$CONFIG"

echo "== done at $(date) =="
