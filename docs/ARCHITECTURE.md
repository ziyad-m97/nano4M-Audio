# Architecture

nano4M-Audio adds audio to the course `nano4M` re-implementation of 4M **without any architectural
change** — the only training-side modification is span masking on the audio stream. This document
records the specifics a reader would need to extend or audit the model.

## Model

| Property | Value |
|---|---|
| Type | encoder–decoder transformer (`nanofm.models.fourm.FourM`) |
| Depth | `enc_depth=6`, `dec_depth=6` (d6-6) |
| Width | `dim=512`, `head_dim=64` |
| Parameters | **95.84 M** (measured; `eval_results/final_summary.json`) |
| Precision | **fp32** (bf16 produced NaNs in the unified 50k-vocab softmax / attention backward) |
| Loss | per-modality length-normalized cross-entropy, averaged (`per_modality_loss_avg=True`) |

## The five modalities (shared vocabulary)

| Modality | Tokenizer | Seq len | Vocab |
|---|---|---|---|
| `tok_rgb@196` | 4M-16k DiVAE | 196 | 16,384 |
| `tok_audio@512` | EnCodec 24 kHz, K=2 RVQ @ 1.5 kbps | 512 | 2,048 |
| `tok_depth@196` | Depth-Anything-V2 → 4M-8k DiVAE | 196 | 8,192 |
| `tok_normal@196` | DSINE → 4M-8k DiVAE | 196 | 8,192 |
| `scene_desc` | GPT-2 BPE | ≤64 | 50,304 |

## Unified vocabulary (`overlap_vocab=True`)

This is the subtle part. `fourm.py` sets

```python
self.vocab_size = max(vocab_sizes)      # = 50,304, NOT sum(vocab_sizes)
self.enc_tok_emb = nn.Embedding(self.vocab_size, dim)
self.to_logits   = nn.Linear(dim, self.vocab_size, bias=False)
```

so every modality indexes into **one** embedding table of size 50,304. Token ids are **not**
shifted per modality (the original 4M `overlap_vocab=False` would shift ids into disjoint ranges; here
that would push RGB/caption ids past `max(vocab_sizes)` and trigger a CUDA index assert). Instead,
**additive modality embeddings** (`enc_mod_emb` / `dec_mod_emb`) plus position embeddings
(`overlap_posembs=True`) disambiguate which stream a token belongs to.

**Audio codebook offset.** EnCodec's two codebooks would collide in id-space, so codebook 2 is offset
by **+1024 on disk** (in `scripts/splits/v5_merge_and_split.py`), giving audio ids in `[0, 2047]` that
preserve the C1/C2 distinction inside the shared table.

## Masking

- **Dirichlet (random) masking** for RGB, depth, normal, caption — standard 4M. `input_alphas` and
  `target_alphas` are uniform (`1.0` each). The per-sample input/target token budgets are sampled
  from `[16, 256]`.
  - The lower bound is **16, not 0**: a tiny total budget let the Dirichlet floor every modality to 0
    input tokens, producing an empty encoder and a NaN in the cross-attention softmax. Raising the
    floor fixed it.
- **Span masking** for `tok_audio@512` (`span_modalities: {tok_audio@512: 2}`). A single contiguous
  span (input span immediately followed by target span) is sampled, **aligned to stride 2** so it
  starts and ends on an EnCodec codebook pair. Because audio is temporal, random masking lets the
  decoder copy an adjacent frame; a contiguous span forces it to predict structure it cannot copy.
  See `nanofm/data/multimodal/masking.py` (`SimpleMultimodalMasking`, `_span_positions`).

## Embeddings

Token embedding + additive **modality** embedding + additive **position** embedding (sinusoidal /
learned per the base nano4M), summed before the encoder/decoder.

## What is unchanged from upstream 4M

The transformer blocks, cross-attention, MaskGIT-style iterative decoding, and the DiVAE
detokenizers are the upstream `apple/ml-4m` / nano4M components. The audio additions are confined to
(1) the EnCodec tokenizer + delay/flatten + cb2 offset (data side) and (2) `span_modalities` in the
masking class (training side).
