# Evaluation results

The **actual JSON outputs** of the reported run — the numbers behind the report and website are
verifiable here without re-running anything.

| File | Contents |
|---|---|
| `final_summary.json` | dataset stats, param count (95.84 M), and the headline results |
| `salvage_probes.json` | audio-only class accuracy top-1/3/5, per class (logit-ranking: 10.4% top-1) |
| `audio2caption.json` | audio→caption accuracy (10.66%) + per-class + per-sample predictions (sequence-decoding) |
| `retrieval.json` | cross-modal retrieval R@1/5/10 over 200 candidates (peak depth→audio R@5 = 4.5%) |
| `external_validation.json` | audio→RGB ImageNet ResNet-50 top-5 hit rate (0%) |
| `memorization_check.json` | train vs test audio-suffix completion (2.95% / 4.13%) |
| `audio_generation_probes.json` | class-conditioned audio RMS/ZCR vs ground truth |
| `forward_vs_ar.json` | single-forward vs autoregressive decoding CE |
| `fig1_ce_drop.json` | per-modality final CE + log-vocab baselines + drop |
| `sanity_check_directions.json` | RGB→depth 11.1%, RGB→normal 18.0%, [D+N+cap]→RGB 0.0% (token acc) |
| `fig7_tokenizer_fidelity.json` | RGB tokenizer PSNR 19.1 dB / SSIM 0.80 |
| `diag_tokenizer.json` | tokenizer diagnostic summary |

**Note on classification variants.** Audio→class is reported two ways: *logit-ranking* (top-1 10.4%,
the headline in `salvage_probes.json`) and *naive sequence-decoding* (top-1 10.7%, in
`audio2caption.json`). They agree globally; per-class they differ (sequence-decoding concentrates on
the `cat`/`cow` attractors).
