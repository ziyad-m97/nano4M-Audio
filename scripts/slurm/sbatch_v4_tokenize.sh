#!/bin/bash
#SBATCH --job-name=v4_tokenize
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=/path/to/nano4M-Audio
#SBATCH --output=/path/to/home/logs/v4_tokenize_%j.out
#SBATCH --error=/path/to/home/logs/v4_tokenize_%j.err

set -euo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi | head -16 || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

# Verify deps + DSINE weights already cached
python -c "
import torch, soundfile, geffnet
from encodec import EncodecModel
from fourm.vq.vqvae import DiVAE
from transformers import pipeline
print('deps OK; cuda?', torch.cuda.is_available())
import os
dsine_w = os.path.expanduser('~/.cache/torch/hub/checkpoints/dsine.pt')
print('DSINE weights cached:', os.path.exists(dsine_w))
"

RAW=/path/to/nano4M-Audio/data/raw_v4_merged
OUT=/path/to/nano4M-Audio/data/tokenized_v4
SCR=/path/to/nano4M-Audio/scripts_v4

mkdir -p "$OUT"

# Smoke first (50 stems via --limit) only if SMOKE=1 env var
LIMIT=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT="--limit 50"
  echo "== SMOKE MODE: $LIMIT =="
fi

for SPLIT in test train; do
  echo ""
  echo "================================="
  echo "  SPLIT: $SPLIT   ${LIMIT:-(full)}"
  echo "================================="
  echo "-- 1/4 audio + scene_desc --"
  python $SCR/v4_01_audio.py  --raw_root $RAW --out_root $OUT --split $SPLIT $LIMIT
  echo "-- 2/4 rgb --"
  python $SCR/v4_02_rgb.py    --raw_root $RAW --out_root $OUT --split $SPLIT $LIMIT
  echo "-- 3/4 depth --"
  python $SCR/v4_03_depth.py  --raw_root $RAW --out_root $OUT --split $SPLIT $LIMIT
  echo "-- 4/4 normal --"
  python $SCR/v4_04_normal.py --raw_root $RAW --out_root $OUT --split $SPLIT $LIMIT
done

echo ""
echo "== verify =="
python $SCR/v4_05_verify.py --raw_root $RAW --out_root $OUT

echo "== done at $(date) =="
