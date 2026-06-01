"""data2 normal tokenization (single frame.jpg tiled K=10).

Per stem:
  - Load frame.jpg
  - DSINE predict normal (per-frame, B=1 to dodge the intrins broadcast bug)
  - Resize to 224×224 if needed, re-unit-norm → [1, 3, 224, 224]
  - 4M normal-8k DiVAE → [1, 196] int32
  - Tile K=10× → [10, 196] int32
  - Save → out_root/{split}/tok_normal@196/{stem}.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from fourm.vq.vqvae import DiVAE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_layout import out_path, atomic_save_npy
from data2_layout import iter_stems_data2, frame_path

K = 10
NUM_TOK = 196
IMG_SIZE = 224
TOK_MODEL = "EPFL-VILAB/4M_tokenizers_normal_8k_224-448"


def load_dsine():
    print("loading DSINE via torch.hub (hugoycj/DSINE-hub) ...", flush=True)
    return torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)


def preprocess_rgb(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, int(round(w * s))),
                      max(IMG_SIZE, int(round(h * s)))),
                     Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    return TF.to_tensor(img)  # [3, H, W] in [0, 1]


def predict_normal_one(predictor, frame_p: Path, device: str) -> torch.Tensor:
    rgb01 = preprocess_rgb(frame_p).unsqueeze(0).to(device)  # [1, 3, H, W]
    with torch.no_grad():
        out = predictor.infer_tensor(rgb01)
    if out.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
    norm = out.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return out / norm  # [1, 3, 224, 224]


def tokenize_then_tile(tok: DiVAE, normal: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = tok.tokenize(normal)
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
    predictor = load_dsine()
    print(f"loading {TOK_MODEL} ...", flush=True)
    tok = DiVAE.from_pretrained(TOK_MODEL).to(device).eval()

    items = list(iter_stems_data2(args.raw_root, args.split))
    if args.limit:
        items = items[: args.limit]
    print(f"stems to process: {len(items)}", flush=True)

    stats = {"done": 0, "skipped": 0, "no_frame": 0, "err": 0}
    for i, (sp, _cls, stem, stem_dir) in enumerate(items, 1):
        out_p = out_path(args.out_root, sp, "tok_normal@196", stem, ".npy")
        if already_done(out_p):
            stats["skipped"] += 1
            continue
        fp = frame_path(stem_dir)
        if not fp.exists():
            stats["no_frame"] += 1
            continue
        try:
            n = predict_normal_one(predictor, fp, device)
            ids = tokenize_then_tile(tok, n)
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
