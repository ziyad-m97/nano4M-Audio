"""data2 depth tokenization (single frame.jpg tiled K=10).

Per stem:
  - Load frame.jpg
  - Depth-Anything-V2-Small predict depth
  - Resize 224×224, normalize [-1, 1] → [1, 1, 224, 224]
  - 4M depth-8k DiVAE → [1, 196] int32
  - Tile K=10× → [10, 196] int32
  - Save → out_root/{split}/tok_depth@196/{stem}.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline as hf_pipeline
from fourm.vq.vqvae import DiVAE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_layout import out_path, atomic_save_npy
from data2_layout import iter_stems_data2, frame_path

K = 10
NUM_TOK = 196
IMG_SIZE = 224
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
TOK_MODEL = "EPFL-VILAB/4M_tokenizers_depth_8k_224-448"


def predict_depth_one(depth_pipe, frame_p: Path) -> torch.Tensor:
    img = Image.open(frame_p).convert("RGB")
    out = depth_pipe([img])[0]
    d = out["predicted_depth"]
    if hasattr(d, "cpu"):
        d = d.detach().to("cpu").float()
    else:
        d = torch.tensor(d, dtype=torch.float32)
    if d.ndim == 3:
        d = d.squeeze(0)
    d = d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    d = F.interpolate(d, size=(IMG_SIZE, IMG_SIZE),
                      mode="bicubic", align_corners=False)
    d_min, d_max = d.min(), d.max()
    if (d_max - d_min) > 1e-8:
        d = 2.0 * (d - d_min) / (d_max - d_min) - 1.0
    else:
        d = torch.zeros_like(d)
    return d  # [1, 1, 224, 224]


def tokenize_then_tile(tok: DiVAE, depth: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = tok.tokenize(depth)
    if isinstance(out, tuple):
        out = out[0]
    ids = out.detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (1, NUM_TOK)
    return np.tile(ids, (K, 1)).astype(np.int32)  # [10, 196]


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
    print(f"device={device}; loading {DEPTH_MODEL} ...", flush=True)
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=0 if device == "cuda" else -1,
    )
    print(f"loading {TOK_MODEL} ...", flush=True)
    tok = DiVAE.from_pretrained(TOK_MODEL).to(device).eval()

    items = list(iter_stems_data2(args.raw_root, args.split))
    if args.limit:
        items = items[: args.limit]
    print(f"stems to process: {len(items)}", flush=True)

    stats = {"done": 0, "skipped": 0, "no_frame": 0, "err": 0}
    for i, (sp, _cls, stem, stem_dir) in enumerate(items, 1):
        out_p = out_path(args.out_root, sp, "tok_depth@196", stem, ".npy")
        if already_done(out_p):
            stats["skipped"] += 1
            continue
        fp = frame_path(stem_dir)
        if not fp.exists():
            stats["no_frame"] += 1
            continue
        try:
            d = predict_depth_one(depth_pipe, fp).to(device)
            ids = tokenize_then_tile(tok, d)
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
