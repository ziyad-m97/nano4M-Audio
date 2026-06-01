# Tokenization

Turns filtered clips into the five aligned token streams. Two coexisting sets:

- **`09–13_*.py`** — the clean, numbered reference tokenizers (single dataset build).
- **`v4_*.py` / `data2_*.py`** — the production scripts used for the two shards that were merged into
  the final `tokenized_v5` (the config comment: *"v4 = Ziyad's 11 classes incl. pigeon; data2 =
  Hassan's cat/pig/cow remapped to v4 names"*). `*_layout.py` define the on-disk layout.

| Modality | Script(s) | Tokenizer | Output |
|---|---|---|---|
| Audio | `10_tok_audio.py`, `v4_01_audio.py`, `data2_03_audio.py` | EnCodec 24 kHz, `set_target_bandwidth(1.5)`, K=2 @ 75 Hz | `[1,2,256]` → flatten 512, cb2 +1024 |
| RGB | `09_tok_rgb.py`, `v4_02_rgb.py`, `data2_04_rgb.py` | 4M-16k DiVAE | 196 tokens, vocab 16,384 |
| Depth | `11_tok_depth.py`, `v4_03_depth.py`, `data2_01_depth.py` | Depth-Anything-V2-Small → 4M-8k DiVAE | 196, vocab 8,192 |
| Normal | `12_tok_normal.py`, `v4_04_normal.py`, `data2_02_normal.py` | DSINE (**per-frame**) → 4M-8k DiVAE | 196, vocab 8,192 |
| Caption | (stored per stem) | `"a photo of a <class>"` → GPT-2 BPE | `scene_desc.json` |

`13_verify.py` / `v4_05_verify.py` sanity-check token shapes and counts. SBATCH launchers are in
[`../slurm/`](../slurm/).

> The audio codebook-2 +1024 offset is applied later, during the merge/split step
> (`../splits/v5_merge_and_split.py`), so ids stay in `[0, 2047]`.
