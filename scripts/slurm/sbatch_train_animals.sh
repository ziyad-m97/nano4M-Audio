#!/bin/bash
#SBATCH --job-name=nano4m_audio
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/home/logs/nano4m_audio_%j.out
#SBATCH --error=/path/to/home/logs/nano4m_audio_%j.err
# Switch to H100 if l40s too slow: change --partition above.

set -euo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

NANO4M_DIR=/path/to/nanofm
CONFIG=$NANO4M_DIR/cfgs/nano4M/animals_audio_d4-4w512.yaml

# Make `import nanofm` resolve to our scratch copy (not the shared install)
export PYTHONPATH=$NANO4M_DIR:${PYTHONPATH:-}

cd "$NANO4M_DIR"
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_training.py --config "$CONFIG"

echo "== done at $(date) =="
