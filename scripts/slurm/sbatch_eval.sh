#!/bin/bash
#SBATCH --job-name=nano4m_eval
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/path/to/nano4M-Audio/eval
#SBATCH --output=/path/to/home/logs/eval_%j.out
#SBATCH --error=/path/to/home/logs/eval_%j.err

set -uo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi | head -16 || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
export PYTHONPATH=/path/to/nanofm:${PYTHONPATH:-}

cd /path/to/nano4M-Audio/eval
python final_evaluation.py

echo "== done at $(date) =="
