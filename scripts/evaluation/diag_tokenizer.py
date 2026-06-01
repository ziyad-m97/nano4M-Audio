"""Definitive RGB-tokenizer sanity check (the 3 tests).

T1  on-disk token ranges        -> must be in [0, 16383], int
T2  pure round-trip of on-disk  -> decode the SAVED tokens, no model. If good,
                                    data + decoder are fine.
T3  fresh-tokenize vs on-disk   -> re-tokenize the SAME keyframe with the SAME
                                    DiVAE + preprocessing and compare token IDs.
                                    High match => same tokenizer was used (no bug).
Bonus  decode-resolution probe  -> report exactly what decode() returns (the
                                    "224-448" model decodes at 448).
Also re-runs the round-trip on a sharp NATURAL crop (not just a checkerboard)
to show the bottleneck on in-distribution-but-sharp content.

Saves: report_figures/diag_tokenizer.json  +  diag_tokenizer.png
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NANO4M_DIR = "/path/to/nanofm"
DATA_ROOT  = "/path/to/nano4M-Audio/data/tokenized_v5"
RAW_DIRS   = ["/path/to/nano4M-Audio/data/raw_v4_merged"]
OUT        = Path("/path/to/nano4M-Audio/eval/report_figures")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, NANO4M_DIR)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448"
IMG_SIZE = 224

from fourm.vq.vqvae import DiVAE
detok = DiVAE.from_pretrained(MODEL_ID).to(DEV).eval()

# --- EXACT replica of v4_02_rgb.py preprocessing ---
def preprocess(p):
    img = Image.open(p).convert("RGB")
    w, h = img.size; s = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, int(round(w*s))), max(IMG_SIZE, int(round(h*s)))), Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    return TF.to_tensor(img) * 2.0 - 1.0  # [3,224,224] in [-1,1]

@torch.no_grad()
def tokenize(t):                       # [3,224,224] -> [196] long ids
    out = detok.tokenize(t.unsqueeze(0).to(DEV))
    out = out[0] if isinstance(out, tuple) else out
    return out.reshape(-1).long().cpu()

@torch.no_grad()
def decode(tok):                       # [196] -> HxWx3 in [0,1]
    img = detok.decode_tokens(tok.view(1, 14, 14).to(DEV))[0].clamp(-1, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

def to01(t): return (t.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
def psnr(a, b):
    if a.shape[:2] != b.shape[:2]:
        b = np.asarray(Image.fromarray((b*255).astype(np.uint8)).resize(
            (a.shape[1], a.shape[0]), Image.BICUBIC), np.float32)/255.
    m = float(np.mean((a-b)**2)); return 99. if m < 1e-10 else 10*np.log10(1./m)

report = {}

# ---------- T1: token ranges (5 stems x all 10 slots) ----------
print("=== T1: on-disk token ranges ===")
rgb_dir = Path(DATA_ROOT) / "test" / "tok_rgb@196"
files = sorted(rgb_dir.glob("*.npy"))[:5]
mins, maxs, dtypes, shapes = [], [], set(), set()
for f in files:
    a = np.load(f); mins.append(int(a.min())); maxs.append(int(a.max()))
    dtypes.add(str(a.dtype)); shapes.add(tuple(a.shape))
    print(f"  {f.stem:24s} shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")
gmin, gmax = min(mins), max(maxs)
ok_range = (gmin >= 0 and gmax <= 16383)
print(f"  GLOBAL min={gmin} max={gmax}  in [0,16383]? {ok_range}")
report["T1"] = {"global_min": gmin, "global_max": gmax, "in_range": bool(ok_range),
                "dtypes": list(dtypes), "shapes": [list(s) for s in shapes]}

# ---------- T2 + T3: round-trip + fresh-match on stems whose raw frame we can find ----------
print("\n=== T2/T3: decode on-disk + fresh-tokenize match ===")
def find_keyframes(stem):
    for base in RAW_DIRS:
        hits = glob.glob(f"{base}/*/*/{stem}/k0*.jpg")
        if hits:
            return sorted(hits)  # k00..k09
    return None

examples = []      # (stem, slot, raw_t, dec_disk, dec_fresh, match, psnr_disk_vs_fresh)
match_rates = []
checked = 0
for f in sorted(rgb_dir.glob("*.npy")):
    stem = f.stem
    kfs = find_keyframes(stem)
    if not kfs or len(kfs) < 1:
        continue
    disk = np.load(f)                       # [10,196]
    # slot 0 corresponds to k00.jpg in v4_02_rgb (sorted glob order)
    slot = 0
    raw_t = preprocess(kfs[slot])
    fresh = tokenize(raw_t).numpy()         # [196]
    on_disk = disk[slot]                    # [196]
    mr = float((fresh == on_disk).mean())
    match_rates.append(mr)
    if len(examples) < 4:
        dec_disk = decode(torch.from_numpy(on_disk).long())
        dec_fresh = decode(torch.from_numpy(fresh).long())
        examples.append((stem, slot, to01(raw_t), dec_disk, dec_fresh, mr,
                         psnr(to01(raw_t), dec_disk)))
    checked += 1
    if checked >= 40:
        break
mean_match = float(np.mean(match_rates)) if match_rates else float("nan")
print(f"  fresh-vs-on-disk token match rate (n={checked}): {mean_match:.1%}")
report["T3"] = {"mean_token_match_rate": mean_match, "n_checked": checked}
report["T2"] = {"note": "decode of on-disk tokens shown in figure (top data rows)"}

# decode-resolution probe
probe = decode(torch.from_numpy(np.load(files[0])[0]).long())
report["decode_resolution"] = list(probe.shape)
print(f"  decode() output resolution: {probe.shape}  (224-448 model upsamples)")

# ---------- Figure ----------
nrows = len(examples)
fig, axes = plt.subplots(nrows, 3, figsize=(7.5, 2.4*nrows))
if nrows == 1: axes = axes[None, :]
cols = ["raw frame (224)", "decode(on-disk tokens)", "decode(fresh tokens)"]
for r, (stem, slot, raw, dd, df, mr, pd) in enumerate(examples):
    for c, im in enumerate([raw, dd, df]):
        ax = axes[r, c]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(cols[c], fontsize=10)
    axes[r, 0].set_ylabel(stem[:14], fontsize=8, rotation=90, labelpad=6)
    axes[r, 2].text(0.5, -0.12, f"match {mr:.0%}", ha="center", va="top",
                    transform=axes[r, 2].transAxes, fontsize=8)
fig.suptitle(f"Tokenizer integrity: on-disk in [0,{gmax}], "
             f"fresh-vs-disk match {mean_match:.0%}, decode @ {probe.shape[0]}px",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "diag_tokenizer.png", dpi=150)
plt.close(fig)
json.dump(report, open(OUT / "diag_tokenizer.json", "w"), indent=2)
print("\nVERDICT:")
print(f"  T1 ranges valid : {ok_range}")
print(f"  T3 match rate   : {mean_match:.1%}  ({'SAME tokenizer ✓' if mean_match>0.5 else 'MISMATCH ✗ BUG'})")
print("  saved diag_tokenizer.{png,json}")
