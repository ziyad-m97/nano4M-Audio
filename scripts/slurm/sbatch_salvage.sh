#!/bin/bash
#SBATCH --job-name=nano4m_salvage
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --chdir=/path/to/nano4M-Audio/eval
#SBATCH --output=/path/to/home/logs/salvage_%j.out
#SBATCH --error=/path/to/home/logs/salvage_%j.err

set -uo pipefail
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
export PYTHONPATH=/path/to/nanofm:${PYTHONPATH:-}
cd /path/to/nano4M-Audio/eval
python salvage_probes.py
echo "== done at $(date) =="
