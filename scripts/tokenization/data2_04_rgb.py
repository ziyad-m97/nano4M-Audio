"""data2 RGB tokenization (single frame.jpg → tile K=10).

Per stem:
  - Load frame.jpg, short-edge resize → 224 center crop → [-1, 1]
  - DiVAE 4M-16k → [1, 196] int32
  - Tile K=10× → [10, 196] int32
  - Write {out}/{split}/tok_rgb@196/as_{stem}.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from fourm.vq.vqvae import DiVAE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_layout import out_path, atomic_save_npy
from data2_layout import iter_stems_data2, frame_path

K = 10
NUM_TOK = 196
IMG_SIZE = 224
MODEL_ID = "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448"


def preprocess(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, int(round(w * s))),
                      max(IMG_SIZE, int(round(h * s)))),
                     Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    return TF.to_tensor(img) * 2.0 - 1.0


def tokenize_one(tok: DiVAE, img: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = tok.tokenize(img.unsqueeze(0))
    if isinstance(out, tuple):
        out = out[0]
    ids = out.detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (1, NUM_TOK)
    return np.tile(ids, (K, 1)).astype(np.int32)


def already_done(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        a = np.load(p)
        return a.shape == (K, NUM_TOK) and a.dtype == np.int32
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading {MODEL_ID} ...", flush=True)
    tok = DiVAE.from_pretrained(MODEL_ID).to(device).eval()

    items = list(iter_stems_data2(args.raw_root, args.split))
    if args.limit:
        items = items[: args.limit]
    print(f"stems to process: {len(items)}", flush=True)

    stats = {"done": 0, "skipped": 0, "no_frame": 0, "err": 0}
    for i, (sp, _cls, stem, stem_dir) in enumerate(items, 1):
        out_p = out_path(args.out_root, sp, "tok_rgb@196", stem, ".npy")
        if already_done(out_p):
            stats["skipped"] += 1
            continue
        fp = frame_path(stem_dir)
        if not fp.exists():
            stats["no_frame"] += 1
            continue
        try:
            ids = tokenize_one(tok, preprocess(fp).to(device))
            atomic_save_npy(out_p, ids)
            stats["done"] += 1
        except Exception as e:
            print(f"  err {stem}: {e}", flush=True)
            stats["err"] += 1
        if i % 100 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] {stats}", flush=True)
    print(f"final: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
