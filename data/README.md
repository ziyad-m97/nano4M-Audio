# Data

**No clips or token files are committed** (3rd-party licensing + size). This directory holds only
the per-clip metadata/manifests; everything else is hosted externally or rebuilt.

## Fast path (graders)

```bash
huggingface-cli download ziyad-m97/nano4m-audio-tokenized --repo-type dataset \
    --local-dir data/tokenized_v5
```
~500 MB; then follow the fast path in the main README.

## Tiers

| Tier | What | Size | Where |
|---|---|---|---|
| 1 | Raw VGGSound / AudioSet clips | ~50–100 GB | not redistributed — see [`../docs/DATASET.md`](../docs/DATASET.md) |
| 2 | Filtered, paired clips | ~5–10 GB | host externally (HuggingFace) |
| 3 | **Tokenized** `tok_rgb/audio/depth/normal` + `scene_desc` | ~500 MB | HuggingFace (`ziyad-m97/nano4m-audio-tokenized`) — what the eval needs |

## `metadata/`

- `final_manifest.csv` — per-clip oracle scores from the 3-stage filter. Columns:
  `youtube_id, start_s, class, source, pann_score, pann_window_offset_s, clip_score,
  clip_best_frame, vad_voiced_frac, audio_path, full_10s_path`. This is the authoritative record of
  which clips were kept and why (PANNs ≥ 0.30, CLIP ≥ 0.25, VAD voiced-frac = 0).
- `manifest_train.csv`, `manifest_test.csv` — the per-split manifests.

The deterministic split (seed 42) is committed at the repo root: [`../splits.json`](../splits.json).

> To upload the tokenized dataset yourself: `huggingface-cli upload ziyad-m97/nano4m-audio-tokenized
> data/tokenized_v5 --repo-type dataset`.
