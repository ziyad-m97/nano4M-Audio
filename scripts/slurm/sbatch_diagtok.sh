#!/bin/bash
#SBATCH --job-name=diagtok
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --chdir=/path/to/nano4M-Audio/eval
#SBATCH --output=/path/to/home/logs/diagtok_%j.out
#SBATCH --error=/path/to/home/logs/diagtok_%j.err
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
export PYTHONPATH=/path/to/nanofm:${PYTHONPATH:-}
python diag_tokenizer.py
