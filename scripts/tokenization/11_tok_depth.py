"""Depth tokenization: Depth-Anything-V2-Small → 4M depth-8k DiVAE.

For each stem and each of the 10 RGB frames:
  - Predict depth with depth-anything/Depth-Anything-V2-Small-hf (HF pipeline)
  - Resize the depth map to 224×224, normalize to [-1, 1], shape [1, 224, 224]
  - Tokenize via EPFL-VILAB/4M_tokenizers_depth_8k_224-448 (DiVAE) → [196] int32
Stack [10, 196] int32 → tokenized/{split}/tok_depth@196/{stem}.npy

⚠️ Do NOT pass RGB directly to the depth tokenizer — it expects 1-channel depth.

Resumable: skips files that already exist with the right shape/dtype.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline as hf_pipeline
from fourm.vq.vqvae import DiVAE

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
CLIPS = ROOT / "clips"
OUT_ROOT = ROOT / "tokenized"

K = 10
NUM_TOK = 196
IMG_SIZE = 224
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
TOK_MODEL = "EPFL-VILAB/4M_tokenizers_depth_8k_224-448"


def load_models(device: str):
    print(f"loading {DEPTH_MODEL} ...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=0 if device == "cuda" else (-1 if device == "cpu" else device),
    )
    print(f"loading {TOK_MODEL} ...")
    tok = DiVAE.from_pretrained(TOK_MODEL).to(device).eval()
    return depth_pipe, tok


def predict_depth_for_frames(depth_pipe, frame_paths: list[Path]
                             ) -> torch.Tensor:
    """Run Depth-Anything on a list of frame paths.

    Returns [K, 1, 224, 224] float tensor in [-1, 1].
    """
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    out = depth_pipe(images)
    depths = []
    for entry in out:
        d = entry["predicted_depth"]  # tensor [H, W]
        if hasattr(d, "cpu"):
            d = d.detach().to("cpu").float()
        else:
            d = torch.tensor(d, dtype=torch.float32)
        if d.ndim == 3:           # some versions return [1, H, W]
            d = d.squeeze(0)
        # Resize to IMG_SIZE × IMG_SIZE with bicubic
        d = d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        d = F.interpolate(d, size=(IMG_SIZE, IMG_SIZE),
                          mode="bicubic", align_corners=False)
        # Normalize per-frame to [-1, 1] using min-max
        d_min, d_max = d.min(), d.max()
        if (d_max - d_min) > 1e-8:
            d = 2.0 * (d - d_min) / (d_max - d_min) - 1.0
        else:
            d = torch.zeros_like(d)
        depths.append(d.squeeze(0))    # [1, H, W]
    return torch.stack(depths, dim=0)  # [K, 1, H, W]


def tokenize_batch(tok: DiVAE, batch: torch.Tensor) -> np.ndarray:
    """[K, 1, 224, 224] in [-1, 1] → [K, 196] int32."""
    with torch.no_grad():
        out = tok.tokenize(batch)
    if isinstance(out, tuple):
        out = out[0]
    ids = out.detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (batch.shape[0], NUM_TOK), \
        f"unexpected tokenize output shape {ids.shape}"
    return ids.astype(np.int32)


def stem_class_pairs(manifest_csv: Path) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(manifest_csv.open()))
    return [(f"{r['youtube_id']}_{int(round(float(r['start_s'])))}", r["class"])
            for r in rows]


def already_done(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        arr = np.load(p)
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

    out_dir = OUT_ROOT / args.split / "tok_depth@196"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    depth_pipe, tok = load_models(device)

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
            depth = predict_depth_for_frames(depth_pipe, paths).to(device)
            ids = tokenize_batch(tok, depth)
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
