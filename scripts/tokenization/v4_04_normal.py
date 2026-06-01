"""v4 normal tokenization: DSINE (torch.hub) → 4M normal-8k DiVAE.

Per stem × 10 keyframes:
  - Predict surface normals with hugoycj/DSINE-hub (3 channels, unit-norm in [-1, 1])
  - Resize to 224×224 if needed → [3, 224, 224]
  - Tokenize via EPFL-VILAB/4M_tokenizers_normal_8k_224-448
  - Save → {out}/{split}/tok_normal@196/{stem}.npy, shape [10, 196] int32

Critical: DO NOT feed RGB directly into the DiVAE — it expects a normal map.
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
from v4_layout import (
    iter_stems, keyframe_paths, out_path, atomic_save_npy,
)

K = 10
NUM_TOK = 196
IMG_SIZE = 224
TOK_MODEL = "EPFL-VILAB/4M_tokenizers_normal_8k_224-448"
def load_dsine(device: str):
    """Load DSINE via torch.hub. Returns a Predictor wrapper (NOT nn.Module);
    the hubconf has the model hard-bound to cuda already, so we don't call
    .to() or .eval() — calling them errors out."""
    print("loading DSINE via torch.hub (hugoycj/DSINE-hub) ...", flush=True)
    return torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)


def preprocess_rgb_for_dsine(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = IMG_SIZE / min(w, h)
    img = img.resize((max(IMG_SIZE, int(round(w * s))),
                      max(IMG_SIZE, int(round(h * s)))),
                     Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    return TF.to_tensor(img)  # [3, H, W] in [0, 1]


def predict_normals(predictor, frame_paths: list[Path], device: str
                    ) -> torch.Tensor:
    """[K, 3, 224, 224] in [-1, 1].

    DSINE Predictor.infer_tensor bakes batch=1 intrinsics inside, so passing
    a [B, 3, H, W] batch breaks an internal tensor concat. Call per-frame.
    """
    rgb01 = torch.stack([preprocess_rgb_for_dsine(p) for p in frame_paths])
    rgb01 = rgb01.to(device)
    preds = []
    with torch.no_grad():
        for i in range(rgb01.shape[0]):
            p = predictor.infer_tensor(rgb01[i:i+1])  # [1, 3, H, W]
            preds.append(p[0])
    out = torch.stack(preds, dim=0)  # [K, 3, H, W]
    if out.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        out = F.interpolate(out, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
    norm = out.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return out / norm


def tokenize(tok: DiVAE, batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = tok.tokenize(batch)
    if isinstance(out, tuple):
        out = out[0]
    ids = out.detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (batch.shape[0], NUM_TOK)
    return ids.astype(np.int32)


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
    ap.add_argument("--class-filter", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dsine = load_dsine(device)
    print(f"loading {TOK_MODEL} ...", flush=True)
    tok = DiVAE.from_pretrained(TOK_MODEL).to(device).eval()

    items = list(iter_stems(args.raw_root, args.split, args.class_filter))
    if args.limit:
        items = items[: args.limit]
    print(f"stems to process: {len(items)}", flush=True)

    stats = {"done": 0, "skipped": 0, "no_frames": 0, "err": 0}
    for i, (sp, cls, stem_dir) in enumerate(items, 1):
        stem = stem_dir.name
        out_p = out_path(args.out_root, sp, "tok_normal@196", stem, ".npy")
        if already_done(out_p):
            stats["skipped"] += 1
            continue
        paths = keyframe_paths(stem_dir)[:K]
        if len(paths) < K:
            stats["no_frames"] += 1
            continue
        try:
            normals = predict_normals(dsine, paths, device)
            ids = tokenize(tok, normals)
            atomic_save_npy(out_p, ids)
            stats["done"] += 1
        except Exception as e:
            print(f"  err {stem}: {e}", flush=True)
            stats["err"] += 1
        if i % 50 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] {stats}", flush=True)
    print(f"final: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
