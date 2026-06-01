#!/bin/bash
#SBATCH --job-name=tokenize_audio_4m
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --chdir=/path/to/animals_dataset
#SBATCH --output=/path/to/home/logs/tokenize_%j.out
#SBATCH --error=/path/to/home/logs/tokenize_%j.err
# To switch to H100 instead (if l40s is too slow):
#   change --partition above to h100 (or whatever the course H100 partition is)

set -euo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

# our extras (already pip-installed --user)
python -c "from encodec import EncodecModel; import soundfile; from fourm.vq.vqvae import DiVAE; from fourm.models.fm import FM; print('deps OK')"

PY=python
cd /path/to/animals_dataset

echo "== 0/5 split (idempotent) =="
$PY scripts/08_split_train_test.py

for SPLIT in test train; do
  echo ""
  echo "================================"
  echo "  SPLIT: $SPLIT"
  echo "================================"
  echo "== 1/4 RGB =="
  $PY scripts/09_tok_rgb.py    --split $SPLIT
  echo "== 2/4 Audio + scene_desc =="
  $PY scripts/10_tok_audio.py  --split $SPLIT
  echo "== 3/4 Depth =="
  $PY scripts/11_tok_depth.py  --split $SPLIT
  echo "== 4/4 Normal =="
  $PY scripts/12_tok_normal.py --split $SPLIT
done

echo ""
echo "== verify =="
$PY scripts/13_verify.py --split both

echo "== done at $(date) =="
