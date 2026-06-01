"""Normal tokenization: 4M-21_B GenerationSampler RGB → tok_normal@224.

⚠️ Do NOT pass RGB directly to the normal DiVAE — it expects a 3-channel
normal map. Instead we follow the team's pattern: use 4M-21 to *generate*
normal tokens from the RGB image conditioning, via MaskGIT 1-step.

For each stem and each of the 10 RGB frames:
  - Preprocess RGB to 224×224 in [-1, 1]
  - Build a generation schedule cond=rgb@224, target=tok_normal@224, 1 step
  - Run sampler, extract the 196 generated token ids
Stack [10, 196] int32 → tokenized/{split}/tok_normal@196/{stem}.npy

Resumable: skips files already done with right shape/dtype.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from fourm.models.fm import FM
from fourm.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from fourm.data.modality_info import MODALITY_INFO

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
CLIPS = ROOT / "clips"
OUT_ROOT = ROOT / "tokenized"

K = 10
NUM_TOK = 196
IMG_SIZE = 224
MODEL_ID = "EPFL-VILAB/4M-21_B"
COND = "rgb@224"
TARGET = "tok_normal@224"


def preprocess_image(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = IMG_SIZE / short
    img = img.resize((max(IMG_SIZE, int(round(w * scale))),
                      max(IMG_SIZE, int(round(h * scale)))),
                     Image.BICUBIC)
    img = TF.center_crop(img, [IMG_SIZE, IMG_SIZE])
    t = TF.to_tensor(img) * 2.0 - 1.0
    return t


def load_sampler(device: str):
    print(f"loading {MODEL_ID} ...")
    model = FM.from_pretrained(MODEL_ID).to(device).eval()
    sampler = GenerationSampler(model)
    return sampler


def build_schedule(device: str, ntoks: int):
    return build_chained_generation_schedules(
        cond_domains=[COND],
        target_domains=[TARGET],
        tokens_per_target=[ntoks],
        autoregression_schemes=["roar"],
        decoding_steps=[1],
        token_decoding_schedules=["linear"],
        temps=[1.0],
        temp_schedules=["constant"],
        cfg_scales=[1.0],
        cfg_schedules=["constant"],
        cfg_grow_conditioning=True,
    )


def generate_normals(sampler: GenerationSampler, schedule,
                     batch_rgb: torch.Tensor, device: str) -> np.ndarray:
    """batch_rgb [K, 3, 224, 224] → [K, 196] int32."""
    bsz = batch_rgb.shape[0]
    batched = {
        COND: {
            "tensor": batch_rgb,
            "input_mask": torch.zeros(bsz, NUM_TOK, dtype=torch.bool,
                                      device=device),
            "target_mask": torch.ones(bsz, NUM_TOK, dtype=torch.bool,
                                      device=device),
        }
    }
    batched = init_empty_target_modality(
        batched, MODALITY_INFO, TARGET, bsz, NUM_TOK, device)
    batched = init_full_input_modality(batched, MODALITY_INFO, COND, device)

    out = sampler.generate(
        batched, schedule, text_tokenizer=None, verbose=False,
        seed=42, top_p=0.0, top_k=0,
    )
    # Extract tokens for our target modality.
    ids = out[TARGET]["tensor"].detach().to("cpu").long().numpy()
    if ids.ndim == 3:
        ids = ids.reshape(ids.shape[0], -1)
    assert ids.shape == (bsz, NUM_TOK), f"got {ids.shape}"
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

    out_dir = OUT_ROOT / args.split / "tok_normal@196"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    sampler = load_sampler(device)
    schedule = build_schedule(device, NUM_TOK)

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
            ids = generate_normals(sampler, schedule, batch, device)
        except Exception as e:
            print(f"  err {stem}: {e}")
            stats["err"] += 1
            continue
        np.save(out_path, ids)
        stats["done"] += 1
        if i % 10 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {stats}")
    print(f"\nfinal: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
