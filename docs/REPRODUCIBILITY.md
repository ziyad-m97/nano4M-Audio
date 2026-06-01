# Reproducibility

From "I just cloned the repo" to "I have the final numbers and figures." Two paths: a **fast path**
(download tokenized data + checkpoint, evaluate — ~15 min) and a **full path** (rebuild everything).

> **Paths in scripts.** Scripts pulled from the cluster have their personal absolute paths replaced
> by neutral placeholders like `/path/to/nano4M-Audio`, `/path/to/nanofm`, `/path/to/scratch`. Set
> them to your locations (or run from the repo root, where the config already uses relative paths).
> The W&B key has been removed — `export WANDB_API_KEY=<your-key>` yourself, or set `log_wandb: False`
> in the config.

## 1. Hardware

- **Training:** 1× NVIDIA H100 80 GB (the reported run: ~1h10 wall-clock, fp32).
- **Eval-only / fast path:** a single GPU with ~16 GB, or CPU for the lighter probes.

## 2. Software

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # installs the local `nanofm` package
```

Python 3.10, CUDA 12.1, PyTorch 2.1. See `requirements.txt` / `environment.yml`. The 4M framework is
pulled from `apple/ml-4m`.

## 3. Data

**Fast path:**
```bash
huggingface-cli download zed-m97/nano4m-audio-tokenized tokenized_v5.tar.gz \
    --repo-type dataset --local-dir data/ && tar xzf data/tokenized_v5.tar.gz -C data/
```
**Full path:** follow [`DATASET.md`](DATASET.md) (download → 3-stage filter → tokenize → split). The
resulting `splits.json` must match the committed one (same seed 42, same data → same partition).

## 4. Checkpoint (fast path) or train (full path)

**Download:**
```bash
huggingface-cli download zed-m97/nano4m-audio checkpoint-final.safetensors \
    --local-dir outputs/animal_full_5mod_v5
```
**Train from scratch:**
```bash
export WANDB_API_KEY=<your-key>          # or set log_wandb: False in the config
torchrun --nproc_per_node=1 run_training.py --config cfgs/nano4M/animal_full_5mod_v5.yaml
```
Final run: **18,311 steps**, batch 64, ~600 M tokens, ~1h10 on 1× H100, fp32. Checkpoints land in
`runs/<run_name>/` (gitignored). On SLURM, see `scripts/slurm/sbatch_train_v5.sh` and the
`scripts/slurm/orchestrate_v5.sh` launcher.

## 5. Evaluate

```bash
jupyter notebook notebooks/final_evaluation.ipynb     # runs all probes end-to-end
# or the scripts directly:
python scripts/evaluation/final_evaluation.py
python scripts/evaluation/make_report_figures.py
python scripts/evaluation/salvage_probes.py
```

Outputs are written to `eval_results/` and `figures/`. The committed versions are the **actual
outputs of the reported run**, so the numbers below should reproduce.

## 6. Expected numbers (from `eval_results/`)

| Probe | Value | Random baseline |
|---|---|---|
| Params | 95.84 M | — |
| Audio eval CE | 5.28 nats | 7.62 (log 2048); ~6.2 empirical marginal |
| Depth / Normal eval CE | 5.11 / 3.45 | 9.01 |
| RGB eval CE | 9.14 | 9.70 |
| Audio→class top-1 / top-5 | 10.4% / 48.4% | 9.1% / 45.5% |
| Audio→class top-1 (seq-decode) | 10.7% | 9.1% |
| Best cross-modal retrieval R@5 (depth→audio) | 4.5% | 2.5% |
| Audio→RGB ImageNet top-5 hit | 0% | ~5% |
| Memorization probe (train / test acc) | 2.95% / 4.13% | — |
| RGB→depth / RGB→normal token acc | 11.1% / 18.0% | ~0.012% (1/8192) |

## 7. Regenerate report figures

See the table in the main `README.md`. `python figures/render_fig1.py` rebuilds the CE-drop figure
from `eval_results/fig1_ce_drop.json`; the galleries come from
`scripts/evaluation/make_report_figures.py`.

## 8. Troubleshooting

- **NaNs during training** → ensure `dtype: fp32`; bf16 is unstable in the 50k-vocab softmax.
- **CUDA index assert in the embedding** → keep `overlap_vocab: True` (the model uses
  `max(vocab_sizes)`, not shifted ranges).
- **NaN in cross-attention on step 1** → keep the input/target token range floor at 16 (empty-encoder
  guard).
- **Corrupted surface normals** → DSINE must be called **per frame** (batch size 1).
- **W&B errors** → `export WANDB_API_KEY=<key>` or set `log_wandb: False`.
- **`splits.json` mismatch** → check you used seed 42 and the same merged token set.
