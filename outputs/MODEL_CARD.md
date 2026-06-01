---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - multimodal
  - audio
  - masked-modeling
  - 4m
  - encodec
  - cross-modal
  - epfl
  - com-304
datasets:
  - zed-m97/nano4m-audio-tokenized
---

# nano4M-Audio — trained checkpoint

**Extending the 4M masked-multimodal framework to audio as a 5th modality** — a controlled study at
small academic scale (COM-304, EPFL, Spring 2026).

| | |
|---|---|
| 💻 Code & docs | https://github.com/ziyad-m97/nano4M-Audio |
| 🌐 Project website | https://ziyad-m97.github.io/nano4M-Audio/ |
| 📊 Tokenized dataset | https://huggingface.co/datasets/zed-m97/nano4m-audio-tokenized |
| 🧾 Config | [`cfgs/nano4M/animal_full_5mod_v5.yaml`](https://github.com/ziyad-m97/nano4M-Audio/blob/main/cfgs/nano4M/animal_full_5mod_v5.yaml) |

## Overview

`nano4M-Audio` adds **audio** to the 4M encoder–decoder transformer **without any architectural
change** — the only training-side modification is contiguous *span masking* on the audio stream.
It is trained jointly on five tokenized modalities (RGB, audio, depth, surface normals, caption) over
a self-built set of animal-vocalization clips. The structural modalities learn strongly and the
iterative generation framework works in the canonical 4M directions; audio acquires conditional
structure at the token level **but does not lift to usable cross-modal audio↔vision generation**. The
contribution is the precise diagnostic of *why*, not a working audio generator — see the report.

## Model details

| Property | Value |
|---|---|
| Architecture | encoder–decoder transformer (`nanofm.models.fourm.FourM`), d6-6w512 |
| Parameters | **95.84 M** |
| Width / heads | `dim=512`, `head_dim=64`, `enc_depth=6`, `dec_depth=6` |
| Precision | fp32 (bf16 NaN'd in the unified-vocab softmax) |
| Vocabulary | unified, `max(vocab_sizes) = 50,304`; modality + position embeddings disambiguate streams |
| Loss | per-modality, length-normalized cross-entropy, averaged |
| Base framework | [`apple/ml-4m`](https://github.com/apple/ml-4m) + the nano4M course re-implementation |

### Modalities

| Modality | Tokenizer | Seq len | Vocab |
|---|---|---|---|
| `tok_rgb@196` | 4M-16k DiVAE | 196 | 16,384 |
| `tok_audio@512` | EnCodec 24 kHz, K=2 RVQ @ 1.5 kbps (delay/flatten, cb2 +1024) | 512 | 2,048 |
| `tok_depth@196` | Depth-Anything-V2 → 4M-8k DiVAE | 196 | 8,192 |
| `tok_normal@196` | DSINE → 4M-8k DiVAE | 196 | 8,192 |
| `scene_desc` | GPT-2 BPE (`"a photo of a <class>"`) | ≤64 | 50,304 |

## Training

- **18,311 steps**, batch size 64, ~600 M tokens, ~1h10 on **1× NVIDIA H100**, fp32.
- Optimizer AdamW (β = 0.9, 0.95), weight decay 0.05, gradient clip 1.0.
- Cosine LR `1e-4 → 1e-6`, 916 warmup steps. Fixed seed; deterministic clip-level split released.
- **Masking:** standard 4M Dirichlet (random) for RGB/depth/normal/caption with per-sample token
  budgets in `[16, 256]`; **contiguous span masking (stride 2)** for audio so the decoder cannot copy
  an adjacent EnCodec frame.

## Dataset

[`zed-m97/nano4m-audio-tokenized`](https://huggingface.co/datasets/zed-m97/nano4m-audio-tokenized) —
9,192 clips over 11 animal classes (cat, chicken, cow, coyote, dog, duck, horse, lion, pig, sheep,
pigeon), sourced from AudioSet + VGGSound and cleaned by a 3-stage PANNs / CLIP / Silero-VAD filter;
clip-level stratified split (seed 42): 7,347 / 907 / 938 train / val / test. Depth and normal targets
are pseudo-labeled (Depth-Anything-V2, DSINE).

## How to use

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from omegaconf import OmegaConf
from hydra.utils import instantiate

# clone the repo for the model code + config first:
#   git clone https://github.com/ziyad-m97/nano4M-Audio && cd nano4M-Audio && pip install -e .
cfg   = OmegaConf.load("cfgs/nano4M/animal_full_5mod_v5.yaml")
model = instantiate(cfg.model_config)
sd    = load_file(hf_hub_download("zed-m97/nano4m-audio", "checkpoint-final.safetensors"))
model.load_state_dict(sd, strict=False)
model.eval()
```

The full evaluation harness is in [`notebooks/final_evaluation.ipynb`](https://github.com/ziyad-m97/nano4M-Audio/blob/main/notebooks/final_evaluation.ipynb);
the actual outputs are committed under [`eval_results/`](https://github.com/ziyad-m97/nano4M-Audio/tree/main/eval_results).

## Evaluation results (held-out 938-clip test set)

| Probe | Model | Random baseline |
|---|---|---|
| Audio eval CE | 5.28 nats | 7.62 (log 2048); ~6.2 empirical marginal |
| Depth / Normal eval CE | 5.11 / 3.45 | 9.01 |
| RGB eval CE | 9.14 | 9.70 |
| Audio → class, top-1 / top-5 | 10.4% / 48.4% | 9.1% / 45.5% |
| Best cross-modal retrieval R@5 (depth→audio) | 4.5% | 2.5% |
| RGB → depth / RGB → normal token acc | 11.1% / 18.0% | ~0.012% |
| Audio → RGB ImageNet ResNet-50 top-5 hit | 0% | ~5% |
| Memorization probe (train / test acc) | 2.95% / 4.13% | — |
| RGB tokenizer fidelity | PSNR 19.1 dB, SSIM 0.80 | — |

**The asymmetry.** Audio captures ~1 nat of conditional structure per token (CE 5.28 < the ~6.2-nat
marginal) and is weakly class-discriminative, yet cross-modal generation mode-collapses. We trace
this to three causes: (1) a train/inference masking mismatch that makes single-source decoding
out-of-distribution; (2) an acoustic-only EnCodec tokenizer with no class semantics; (3) operating at
~10⁴ clips, below the cross-modal emergence threshold of the contrastive audio-visual literature.

## Intended uses & limitations

- **Intended:** research / educational reference for adding a temporal modality to a 4M-style model,
  and a reproducible small-scale baseline. Not intended for production audio generation.
- **Limitations:** cross-modal audio↔vision *generation* does not work at this scale; single random
  seed; no human listening evaluation; depth/normal labels are pseudo-labels; the audio token budget
  (512 vs 196 RGB) traded visual fidelity for audio context that yielded no measurable cross-modal
  gain. See the report's limitations section.

## Citation

```bibtex
@misc{nano4m-audio-2026,
  author      = {Mellal, Ziyad and Baddour, Hassan and Farhat, Marc},
  title       = {Nano4M-Audio: Adding Audio as a 5th Modality to the 4M Architecture},
  year        = {2026},
  institution = {EPFL, COM-304},
  url         = {https://github.com/ziyad-m97/nano4M-Audio}
}
```

## Acknowledgements

Supervised by **Jason Toskov** at EPFL VILAB. Built on the open-source 4M codebase
([apple/ml-4m](https://github.com/apple/ml-4m); Mizrahi et al., NeurIPS 2023; Bachmann et al.,
NeurIPS 2024). Compute provided by the EPFL SCITAS Kuma cluster.

## License

MIT for the model weights and this repository's contributions. The underlying 4M code is licensed
under Apache-2.0.
