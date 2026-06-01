"""data2 audio + scene_desc tokenization (single audio.wav → tile K=10).

Per stem:
  - Load audio.wav (24 kHz mono, ~81920 samples)
  - EnCodec encode @ 1.5 kbps → codes [K=2, T=256], interleave → 512 ids
  - Tile K=10× → [10, 512] int32 (matching v4 format)
  - Write {out}/{split}/tok_audio@512/as_{stem}.npy
  - Write {out}/{split}/scene_desc/as_{stem}.json = ["<canonical_class>"] * 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_layout import out_path, atomic_save_npy, atomic_write_text
from data2_layout import iter_stems_data2, audio_path

K = 10
EXPECT_SAMPLES = 81920
EXPECT_FRAMES = 256
EXPECT_CODEBOOKS = 2
TOK_PER_SLOT = EXPECT_FRAMES * EXPECT_CODEBOOKS  # 512


def load_audio(wav_path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if sr != 24000:
        raise RuntimeError(f"{wav_path}: expected 24kHz, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < EXPECT_SAMPLES:
        audio = np.pad(audio, (0, EXPECT_SAMPLES - len(audio)))
    elif len(audio) > EXPECT_SAMPLES:
        audio = audio[:EXPECT_SAMPLES]
    return torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)


def encode(model: EncodecModel, wav: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        frames = model.encode(wav)
    codes = torch.cat([f[0] for f in frames], dim=-1)
    arr = codes[0].cpu().numpy()
    interleaved = arr.T.reshape(-1)
    assert interleaved.shape[0] == TOK_PER_SLOT
    return np.tile(interleaved[None, :], (K, 1)).astype(np.int32)


def already_done_audio(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        a = np.load(p)
        return a.shape == (K, TOK_PER_SLOT) and a.dtype == np.int32
    except Exception:
        return False


def already_done_desc(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        caps = json.loads(p.read_text())
        return isinstance(caps, list) and len(caps) == K
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("loading EnCodec 24kHz @ 1.5 kbps ...", flush=True)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)
    model = model.to("cpu").eval()

    items = list(iter_stems_data2(args.raw_root, args.split))
    if args.limit:
        items = items[: args.limit]
    print(f"stems to process: {len(items)}", flush=True)

    stats = {"done_audio": 0, "done_desc": 0, "skipped": 0,
             "no_audio": 0, "err": 0}
    for i, (sp, full_cls, stem, stem_dir) in enumerate(items, 1):
        a_out = out_path(args.out_root, sp, "tok_audio@512", stem, ".npy")
        d_out = out_path(args.out_root, sp, "scene_desc", stem, ".json")

        if not already_done_desc(d_out):
            atomic_write_text(d_out, json.dumps([full_cls] * K))
            stats["done_desc"] += 1

        if already_done_audio(a_out):
            stats["skipped"] += 1
        else:
            wav = audio_path(stem_dir)
            if not wav.exists():
                stats["no_audio"] += 1
            else:
                try:
                    tokens = encode(model, load_audio(wav))
                    atomic_save_npy(a_out, tokens)
                    stats["done_audio"] += 1
                except Exception as e:
                    print(f"  err {stem}: {e}", flush=True)
                    stats["err"] += 1
        if i % 200 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] {stats}", flush=True)
    print(f"final: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
