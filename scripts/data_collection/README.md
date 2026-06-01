# Data collection

Acquisition of raw animal-vocalization clips from **VGGSound** and **AudioSet**, then a three-stage
cleaning filter. See [`../../docs/DATASET.md`](../../docs/DATASET.md) for the full procedure and
thresholds.

- `pull_pigeon_audioset.py` — example per-class AudioSet puller (segment download + windowing).

## Three-stage filter

| Stage | Oracle | Keep if |
|---|---|---|
| 1 | PANNs (Cnn14) audio tagging | class score ≥ 0.30 |
| 2 | CLIP image–text cosine (best of 10 frames) | ≥ 0.25 |
| 3 | Silero VAD | voiced fraction = 0 (no human speech) |

> The interactive download + filter steps were **not preserved as standalone numbered scripts**. Their
> output — the per-clip `pann_score` / `clip_score` / `vad_voiced_frac` values and source ids — is
> recorded in [`../../data/metadata/final_manifest.csv`](../../data/metadata/final_manifest.csv),
> which together with the source datasets fully determines the kept clip set.
