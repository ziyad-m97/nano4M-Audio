# nano4M-Audio — project website

Publication-style project page for **Nano4M-Audio: Adding Audio as a 5th Modality to the 4M
Architecture** (COM-304, Foundation Models, EPFL, Spring 2026), modeled on
[4m.epfl.ch](https://4m.epfl.ch).

It is a **static site** — plain `index.html` + CSS + a little vanilla JS, no build step, works
offline. Source content tracks the 4-page report (`docs/assets/report.pdf`).

> **Live URL (set in repo settings):** https://ziyad-m97.github.io/nano4M-Audio

---

## Preview locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

(Opening `docs/index.html` directly mostly works, but a local server is needed for the
clipboard/`fetch`-style features and correct MIME types.)

## Deploy on GitHub Pages

1. Push this repository to GitHub.
2. **Settings → Pages → Build and deployment → Source: “Deploy from a branch”.**
3. Branch **`main`**, folder **`/docs`**, Save.
4. The site publishes at `https://<user>.github.io/<repo>` within a minute or two.

To serve from a `gh-pages` branch instead, copy the contents of `docs/` to the root of that branch.

---

## Repository layout

```
docs/
  index.html               # the whole site (10 sections)
  assets/
    css/style.css           # theme + responsive layout
    js/main.js              # scrollspy, mobile nav, copy-BibTeX
    img/                    # diagrams (SVG, final) + figure placeholders
    audio/                  # gt_*.ogg/.mp3, gen_*.ogg/.mp3 (PLACEHOLDER tones)
    report.pdf              # the 4-page report  (real, wired in)
    slides.pdf              # the pitch deck      (real, wired in)
```

### Diagrams (final, hand-authored SVG — no action needed)
`method.svg` · `audio_pipeline.svg` · `dataset_pipeline.svg` · `bidirectional_problem.svg`
· `audio_ce_vs_marginal.svg`

### Figures that auto-swap (`onerror` fallback)
The HTML references these PNGs; until the file exists, a labeled SVG placeholder shows. **Drop the
real PNG at the same path and it appears automatically — no HTML edit needed.**

| Reference (put your PNG here)              | Fallback placeholder shown until then       |
|--------------------------------------------|---------------------------------------------|
| `assets/img/training_curves.png`           | `ph_training_curves.svg`                    |
| `assets/img/reconstruction_gallery.png`    | `ph_reconstruction_gallery.svg`             |
| `assets/img/caption2rgb_gallery.png`       | `ph_caption2rgb_gallery.svg`                |

### Audio (replace the synthetic placeholders)
`assets/audio/` currently holds **synthetic tones** so the player layout works and the demo even
illustrates the mode collapse (varied GT, one flat tone for every generated cell). Replace with the
real EnCodec-decoded clips, keeping the filenames:

```bash
# WAV → OGG (Vorbis, 96 kbps) and MP3 fallback, for each of: dog cat pig sheep chicken horse
ffmpeg -i gt_dog.wav  -c:a libvorbis -b:a 96k -ac 1 gt_dog.ogg
ffmpeg -i gt_dog.wav  -c:a libmp3lame -b:a 96k -ac 1 gt_dog.mp3
# …repeat for gen_*.wav
```

---

## Before submission — fill these in

- [ ] **Author SCIPER numbers** (`index.html`, `SCIPER 000000` ×3).
- [ ] **Role-to-name mapping** in the Team cards — currently assigned in listed order
      (Ziyad→data, Hassan→training, Marc→eval); confirm against the report's Individual
      Contributions section.
- [ ] **Author social links** (currently `href="#"`).
- [x] **GitHub repo URL** — set to `https://github.com/ziyad-m97/nano4M-Audio` in the hero “Code”
      button and the BibTeX `url`. Change only if you move the repo to another account.
- [ ] **caption→RGB ResNet-50 top-5 hit rate** — replace the highlighted `XX%` (appears **twice** in
      the Cross-modal Generation section: the figure caption and the contrast callout) with the
      number from `eval_results/sanity_check_directions.json`.
- [ ] Drop in the three real figure PNGs and the real audio clips (see above).
- [ ] Update the footer date if needed.

---

## Numbers reconciled with the report

The site uses the **report (`main_final.tex`) as the source of truth** wherever it disagreed with
the original website brief:

| Claim                | Brief said      | Report / site says                    |
|----------------------|-----------------|---------------------------------------|
| Parameters           | ~86M            | **~96M** (95.8M, d6-6w512)            |
| Dataset size         | “~11k clips”    | **9,192 clips** (7,347/907/938)      |
| Precision / runtime  | bf16, ~3h       | **fp32, ~1h10** (bf16 NaN’d)         |
| Diagnostic causes    | two             | **three** (masking · tokenizer · scale) |
| Data-scale framing   | ~10× gap        | 10⁴ clips; ~1000× below 4M; 10⁵–10⁶ in the contrastive AV literature |

## Notes

- No tracking, no autoplay, no external CDNs — all assets are local.
- Total page weight is well under the 15 MB budget (≈3 MB, dominated by `slides.pdf`).
- Website based on the [Nerfies template](https://github.com/nerfies/nerfies.github.io),
  Creative Commons Attribution-ShareAlike 4.0 — see `LICENSE`.
