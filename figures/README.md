# Figures

The actual report figures from the reported run (compressed for size). Underlying numbers are in
[`../eval_results/`](../eval_results/).

| File | Report use | Regenerate |
|---|---|---|
| `per_modality_ce_drop.png` | per-modality CE drop bar | `python figures/render_fig1.py` (← `eval_results/fig1_ce_drop.json`) |
| `reconstruction_gallery.png` | depth/normal reconstruction (the modalities that worked) | `scripts/evaluation/make_report_figures.py` |
| `caption2rgb_gallery.png` | caption→RGB generation | `scripts/evaluation/make_report_figures.py` |
| `sanity_check_directions.png` | framework-validation directions (RGB→depth/normal etc.) | `make_report_figures.py` (← `eval_results/sanity_check_directions.json`) |
| `appendix/audio2rgb_grid.png` | audio→RGB generation collapse (0% recognized) | `make_report_figures.py` |
| `appendix/audio2caption_confusion.png` | audio→caption confusion (attractor classes) | `make_report_figures.py` |
| `appendix/class_conditioned_spectrograms.png` | class-conditioned audio collapse | `make_report_figures.py` |
| `appendix/tokenizer_fidelity.png` / `tokenizer_diagnostic.png` | RGB tokenizer PSNR/SSIM (19.1 dB / 0.80) | `diag_tokenizer.py` |
| `appendix/retrieval_barchart.png` | cross-modal retrieval (≤4.5% R@5) | `make_report_figures.py` |

The architecture diagram and the audio-CE-vs-marginal chart are hand-authored SVGs in
[`../docs/assets/img/`](../docs/assets/img/) (`method.svg`, `audio_ce_vs_marginal.svg`). The TikZ
source for the report's architecture figure lives in the report LaTeX project, not this repo.
