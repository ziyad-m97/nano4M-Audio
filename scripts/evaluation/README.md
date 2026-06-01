# Evaluation

The full evaluation suite. Outputs land in [`../../eval_results/`](../../eval_results/) (JSON) and
[`../../figures/`](../../figures/) (PNG/PDF) — the committed versions are the **actual outputs of the
reported run**.

| Script | Produces |
|---|---|
| `final_evaluation.py` | per-modality CE, audio→caption classification, cross-modal retrieval, conditional generation, external (ImageNet ResNet-50) validation, the memorization probe, forward-vs-AR — i.e. the JSONs in `eval_results/` |
| `make_report_figures.py` | the report figure galleries (reconstruction, caption→RGB, audio→RGB grid, sanity-check directions, spectrograms, confusion, retrieval) |
| `salvage_probes.py` | per-class audio-only top-1/3/5 with logit-ranking (`salvage_probes.json`) |
| `diag_tokenizer.py` | tokenizer-fidelity diagnostic (PSNR/SSIM of real vs synthetic token reconstructions) |

The notebook [`../../notebooks/final_evaluation.ipynb`](../../notebooks/final_evaluation.ipynb) runs
the same probes end-to-end given a checkpoint. SBATCH launchers are in [`../slurm/`](../slurm/).

Baselines are explicit in every metric: chance = 1/11 = 9.1% (classification), `k/200` (retrieval),
`log(vocab)` (token CE), ~0.5–5% top-5 (ImageNet).
