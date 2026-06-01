#!/usr/bin/env bash
# Run the full 4M tokenization pipeline (5 modalities) on train + test.
#
# Usage:
#   bash run_tokenize.sh                 # full run
#   bash run_tokenize.sh --smoke 5       # smoke run, 5 clips per split
#   bash run_tokenize.sh --split test    # one split only
#
# Order matches the friend's spec: RGB → Audio → Depth → Normal → Verify.
# Every script is resumable (skips files already produced with right shape).

set -euo pipefail

PY=/Users/marcops/miniconda3/envs/nano4m_data/bin/python
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIMIT_ARG=""
SPLITS=(train test)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) LIMIT_ARG="--limit ${2:-5}"; shift 2 ;;
    --split) SPLITS=("$2"); shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

echo "=== 0/6 split train/test (idempotent) ==="
$PY scripts/08_split_train_test.py

for SPLIT in "${SPLITS[@]}"; do
  echo ""
  echo "============================================"
  echo "  SPLIT: $SPLIT $LIMIT_ARG"
  echo "============================================"
  echo "=== 1/5 RGB (4M-16k DiVAE) ==="
  $PY scripts/09_tok_rgb.py --split "$SPLIT" $LIMIT_ARG
  echo "=== 2/5 Audio (EnCodec 24kHz) + scene_desc ==="
  $PY scripts/10_tok_audio.py --split "$SPLIT" $LIMIT_ARG
  echo "=== 3/5 Depth (Depth-Anything-V2 → 4M depth-8k) ==="
  $PY scripts/11_tok_depth.py --split "$SPLIT" $LIMIT_ARG
  echo "=== 4/5 Normal (4M-21 GenerationSampler) ==="
  $PY scripts/12_tok_normal.py --split "$SPLIT" $LIMIT_ARG
done

echo ""
echo "=== 5/5 verify ==="
$PY scripts/13_verify.py --split both
