# Dataset

A self-built dataset of **9,192** animal-vocalization clips paired with video keyframes, across **11
classes**, drawn from AudioSet and VGGSound and cleaned by a three-stage oracle.

> **No clips or tokens are committed.** This document explains how to obtain/rebuild them. The
> per-clip oracle scores are recorded in [`../data/metadata/`](../data/metadata/), and the
> deterministic split is committed as [`../splits.json`](../splits.json).

## Fast path (graders)

Download the pre-tokenized dataset and skip to evaluation (see the main README):

```bash
huggingface-cli download zed-m97/nano4m-audio-tokenized tokenized_v5.tar.gz \
    --repo-type dataset --local-dir data/ && tar xzf data/tokenized_v5.tar.gz -C data/
```

## Classes

`cat`, `chicken`, `cow`, `coyote`, `dog`, `duck`, `horse`, `lion`, `pig`, `sheep`, `pigeon`
(canonical caption forms: "cat meowing", "dog barking", "lions roaring", …). Each has a distinctive
acoustic signature and a visually identifiable subject, giving naturally paired clips and a clean
classification oracle.

## Full pipeline

### 1 — Acquire raw clips
- **VGGSound** — https://www.robots.ox.ac.uk/~vgg/data/vggsound/
- **AudioSet** — https://research.google.com/audioset/

Clips are 10 s; we keep a 3.413 s (81,920-sample @ 24 kHz) window. `scripts/data_collection/pull_pigeon_audioset.py` is an example per-class AudioSet puller.

### 2 — Three-stage filter
Each candidate clip must pass all three oracles. The thresholds and the per-clip scores are recorded
as columns in `data/metadata/final_manifest.csv`:

| Stage | Oracle | Keep if | Manifest column |
|---|---|---|---|
| 1 | PANNs audio tagging (Cnn14) | class score **≥ 0.30** | `pann_score`, `pann_window_offset_s` |
| 2 | CLIP image–text cosine (best of 10 frames) | **≥ 0.25** | `clip_score`, `clip_best_frame` |
| 3 | Silero VAD | voiced fraction **= 0** (no human speech) | `vad_voiced_frac` |

> The download + filter stages were run interactively during the project and were **not preserved as
> standalone numbered scripts**; the manifest (with the per-clip `pann_score` / `clip_score` /
> `vad_voiced_frac` values above) is the authoritative artifact of this stage and is sufficient to
> reproduce the exact clip set together with the source dataset ids (`youtube_id`, `start_s`).

### 3 — Pseudo-labels & tokenization (5 modalities)
`scripts/tokenization/` (numbered `09–13`, plus the production `v4_*`/`data2_*` variants):

- **RGB** → 4M-16k DiVAE (`10_…`/`v4_02`), 196 tokens, 14×14, vocab 16,384.
- **Audio** → EnCodec 24 kHz, `set_target_bandwidth(1.5)`, K=2 codebooks @ 75 Hz → `[1,2,256]`
  codes, delay/flattened to 512 tokens, codebook 2 offset +1024 (`10_tok_audio`/`v4_01`).
- **Depth** → Depth-Anything-V2-Small → 4M-8k DiVAE (`11_…`/`v4_03`).
- **Normal** → DSINE (**called per-frame** — batched calls bake in batch-1 intrinsics and corrupt the
  normals) → 4M-8k DiVAE (`12_…`/`v4_04`).
- **Caption** → `"a photo of a <class>"`, stored as a per-stem `scene_desc.json` (10 entries, one per
  keyframe) and GPT-2 BPE-tokenized in the dataloader.

Each clip carries **K=10 keyframes**, decoupling visual diversity from audio uniqueness; one random
frame per stem is drawn per `__getitem__` (`sample_from_k_augmentations: 10`).

### 4 — Deterministic split
`scripts/splits/` builds a **clip-level** split stratified by `class × source` with **seed 42**
(`08_split_train_test.py` for the base set; `v5_merge_and_split.py` merges the two tokenized shards,
applies the audio cb2 offset, and writes the committed `splits.json`).

```bash
python scripts/splits/v5_merge_and_split.py \
    --sources data/tokenized_v4 data/tokenized_data2 \
    --out_root data/tokenized_v5 --report splits.json
```

## Statistics

| Split | Stems |
|---|---|
| Train | 7,347 |
| Val | 907 |
| Test | 938 |
| **Total** | **9,192** |

- 11 classes · 10 keyframes/clip · clip-level stratification by (class × source) · seed 42.
- `tokenizer fidelity` on our web imagery: mean PSNR **19.1 dB**, SSIM **0.80** over 60 test images
  (`eval_results/fig7_tokenizer_fidelity.json`) — a tokenizer domain shift that affects pixel-space
  decoding but not the token-space cross-modal metrics.
