#!/bin/bash
#SBATCH --job-name=data2_5mod
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --chdir=/path/to/nano4M-Audio
#SBATCH --output=/path/to/home/logs/data2_5mod_%j.out
#SBATCH --error=/path/to/home/logs/data2_5mod_%j.err

set -euo pipefail
echo "== job $SLURM_JOB_ID on $(hostname) at $(date) =="
nvidia-smi | head -16 || true

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

RAW=/path/to/nano4M-Audio/data/_data2_staging/data
OUT=/path/to/nano4M-Audio/data/tokenized_data2
SCR=/path/to/nano4M-Audio/scripts_v4

mkdir -p "$OUT"

for SPLIT in test train; do
  echo ""
  echo "================================="
  echo "  SPLIT: $SPLIT"
  echo "================================="
  echo "-- 1/4 audio + scene_desc --"
  python $SCR/data2_03_audio.py  --raw_root $RAW --out_root $OUT --split $SPLIT
  echo "-- 2/4 rgb --"
  python $SCR/data2_04_rgb.py    --raw_root $RAW --out_root $OUT --split $SPLIT
  echo "-- 3/4 depth --"
  python $SCR/data2_01_depth.py  --raw_root $RAW --out_root $OUT --split $SPLIT
  echo "-- 4/4 normal --"
  python $SCR/data2_02_normal.py --raw_root $RAW --out_root $OUT --split $SPLIT
done

echo "== done at $(date) =="
