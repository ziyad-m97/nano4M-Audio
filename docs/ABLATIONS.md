# Ablations & engineering decisions

Five decisions shaped the final run, each reported as a self-baseline against the simpler
alternative on the metric it most affects. Numbers are from the final run history and
`eval_results/` where available.

| # | Decision | Change | Impact | Verdict |
|---|---|---|---|---|
| 1 | RGB tokenizer | Cosmos-64k → **4M-16k DiVAE** | RGB CE plateau 10.6 → **9.14** nats (≈1.4 lower) | kept |
| 2 | Audio window | 256 → **512 tokens** | audio CE 6.5 → **5.28** nats (largest single audio gain) | kept |
| 3 | Keyframes / clip | 1 → **10** | 10× visual training signal, audio uniqueness fixed | kept |
| 4 | Oracle thresholds | PANNs + CLIP tightened | dataset purity ≈×2 at ≈½ the size | kept |
| 5 | Model size | d5-w384 / **d6-w512** / d8-w640 | d6-w512 is the sweet spot (d8 over-fits, d5 under-fits) | kept d6 |

### Supporting fixes (kept)

- **Per-modality loss averaging** (`per_modality_loss_avg=True`): length-normalizes CE per modality
  so the 512-token audio sequence does not dominate the ≤64-token caption.
- **fp32 over bf16**: bf16 produced NaNs in the unified 50,304-vocab softmax / attention backward;
  fp32 is stable and still fast on H100 at ~96M params.
- **Empty-encoder guard**: input/target token ranges raised to `[16, 256]` so Dirichlet budgeting
  cannot floor every modality to 0 input tokens (which yielded an empty encoder → NaN cross-attn).
- **`overlap_vocab=True`**: the model uses `vocab_size = max(vocab_sizes)` with additive modality
  embeddings rather than shifting ids into disjoint ranges (shifting would index past the table and
  trigger a CUDA assert). The audio cb2 +1024 offset is applied on disk instead.

### The audio-duration result, in context

Decision 2 (the only tweak that targets audio directly) yields the largest single audio-CE
improvement — consistent with the diagnostic that audio learning is **data- and tokenizer-bound**,
not architecture-bound. In hindsight, spending 512 tokens on audio vs 196 on RGB traded visual
fidelity for audio context that did not produce a measurable **cross-modal** benefit; future work
should ablate that trade-off explicitly.
