"""RGB tokenization with the EPFL-VILAB 4M-16k DiVAE.

For each stem:
  - Load frames/00..09.jpg (the 10 keyframes ffmpeg already extracted)
  - Short-edge resize → 224 center crop → [-1, 1] tensor
  - Tokenize each via DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448")
  - Stack [10, 196] int32 → tokenized/{split}/tok_rgb@196/{stem}.npy

Resumable: skips files that already exist with the right shape/dtype.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from fourm.vq.vqvae import DiVAE

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
CLIPS = ROOT / "clips"
OUT_ROOT = ROOT / "tokenized"

K = 10
NUM_TOK = 196      # 14×14 patches at 224 / patch 16
RGB_MODEL = "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448"
IMG_SIZE = 224


def load_rgb_tokenizer(device: str) -> DiVAE:
    tok = DiVAE.from_pretrained(RGB_MODEL)
    tok = tok.to(device).eval()
    return tok


def preprocess_image(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = IMG_SIZE / short
    img = img.resize((max(IMG_SIZE, int(round(w * scale))),
                      max(IMG_SIZE, int(round(h * scale)))),
                     Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    t = TF.to_tensor(img)               # [3, 224, 224] in [0, 1]
    t = t * 2.0 - 1.0                   # [-1, 1]
    return t


def tokenize_batch(tok: DiVAE, batch: torch.Tensor) -> np.ndarray:
    """batch [K, 3, 224, 224] → [K, 196] int32 token ids."""
    with torch.no_grad():
        out = tok.tokenize(batch)
    if isinstance(out, tuple):
        out = out[0]
    ids = out.detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        # Some DiVAE variants return [K, H, W]; flatten.
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (batch.shape[0], NUM_TOK), \
        f"unexpected tokenize output shape {ids.shape}"
    return ids.astype(np.int32)


def stem_class_pairs(manifest_csv: Path) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(manifest_csv.open()))
    return [(f"{r['youtube_id']}_{int(round(float(r['start_s'])))}", r["class"])
            for r in rows]


def already_done(npy_path: Path) -> bool:
    if not npy_path.exists():
        return False
    try:
        arr = np.load(npy_path)
        return arr.shape == (K, NUM_TOK) and arr.dtype == np.int32
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = META / f"manifest_{args.split}.csv"
    pairs = stem_class_pairs(manifest)
    if args.limit:
        pairs = pairs[: args.limit]

    out_dir = OUT_ROOT / args.split / "tok_rgb@196"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; loading {RGB_MODEL}...")
    tok = load_rgb_tokenizer(device)

    stats = {"done": 0, "skipped": 0, "no_frames": 0, "err": 0}
    for i, (stem, cls) in enumerate(pairs, 1):
        out_path = out_dir / f"{stem}.npy"
        if already_done(out_path):
            stats["skipped"] += 1
            continue
        frames_dir = CLIPS / cls / stem / "frames"
        paths = sorted(frames_dir.glob("*.jpg"))[:K]
        if len(paths) < K:
            stats["no_frames"] += 1
            continue
        try:
            batch = torch.stack([preprocess_image(p) for p in paths]).to(device)
            ids = tokenize_batch(tok, batch)
        except Exception as e:
            print(f"  err {stem}: {e}")
            stats["err"] += 1
            continue
        np.save(out_path, ids)
        stats["done"] += 1
        if i % 25 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {stats}")
    print(f"\nfinal: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
