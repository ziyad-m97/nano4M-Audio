#!/bin/bash
#SBATCH --job-name=execute_nb
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --chdir=/path/to/nano4M-Audio/eval
#SBATCH --output=/path/to/home/logs/execute_nb_%j.out
#SBATCH --error=/path/to/home/logs/execute_nb_%j.err

set -uo pipefail
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
pip install --user --quiet ipykernel 2>&1 | tail -2 || true

export PYTHONPATH=/path/to/nanofm:${PYTHONPATH:-}
cd /path/to/nano4M-Audio/eval

# Clean prior outputs so executed cells produce fresh artifacts
rm -rf eval_out/figures/audio_conditioned_samples eval_out/figures/audio_demos
mkdir -p eval_out/figures eval_out/eval_results

# Execute the notebook, preserving outputs in-place
jupyter nbconvert --to notebook --execute final_evaluation.ipynb \
    --output final_evaluation_executed.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    2>&1 | tail -30

echo "== done at $(date) =="
ls -lh final_evaluation_executed.ipynb 2>/dev/null
