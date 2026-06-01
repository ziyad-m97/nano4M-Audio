"""Audio tokenization with EnCodec 24kHz (K=2 codebooks @ 1.5 kbps).

For each stem in the split manifest:
  - Load audio.wav (24 kHz mono PCM16, 81920 samples = 3.413s)
  - Encode with EncodecModel.encodec_model_24khz at bandwidth 1.5 kbps
  - Get codes shape [1, 2, 256] (B=1, K=2 codebooks, T=256 frames @ 75Hz)
  - Flatten interleaved: [C1[0], C2[0], C1[1], C2[1], ...] → length 512
  - Replicate across 10 K-augmentation slots → [10, 512] int16
  - Save → tokenized/{split}/tok_audio@512/{stem}.npy
  - Write tokenized/{split}/scene_desc/{stem}.json = ["<class>"] * 10

Resumable: skips files that already exist with the right shape/dtype.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
CLIPS = ROOT / "clips"
OUT_ROOT = ROOT / "tokenized"

K = 10                 # K-augmentation slots
EXPECT_SAMPLES = 81920 # 3.413333s @ 24 kHz
EXPECT_FRAMES = 256    # 81920 / 320 stride
EXPECT_CODEBOOKS = 2   # 1.5 kbps @ 24 kHz = 2 codebooks
TOK_PER_SLOT = EXPECT_FRAMES * EXPECT_CODEBOOKS  # 512


def load_audio_for_encodec(wav_path: Path) -> torch.Tensor:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if sr != 24000:
        raise RuntimeError(f"{wav_path}: expected 24kHz, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Pad/trim to exactly 81920 samples so we hit 256 frames.
    if len(audio) < EXPECT_SAMPLES:
        audio = np.pad(audio, (0, EXPECT_SAMPLES - len(audio)))
    elif len(audio) > EXPECT_SAMPLES:
        audio = audio[:EXPECT_SAMPLES]
    return torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # [1, 1, T]


def encode_to_tokens(model: EncodecModel, wav: torch.Tensor) -> np.ndarray:
    """Return [10, 512] int16 with interleaved C1,C2 codes replicated K times."""
    with torch.no_grad():
        frames = model.encode(wav)
    # frames is a list of (codes, scale); for 24kHz single chunk we get 1 entry.
    codes_list = [f[0] for f in frames]
    codes = torch.cat(codes_list, dim=-1)  # [1, K_cb, T]
    assert codes.shape[1] == EXPECT_CODEBOOKS, f"codebooks {codes.shape[1]}"
    assert codes.shape[2] == EXPECT_FRAMES, f"frames {codes.shape[2]}"
    arr = codes[0].cpu().numpy()  # [K_cb=2, T=256]
    # Interleave: [C1[0], C2[0], C1[1], C2[1], ...] → length 512
    interleaved = arr.T.reshape(-1)  # T transpose → [T, K_cb] then flatten
    assert interleaved.shape[0] == TOK_PER_SLOT
    out = np.tile(interleaved[None, :], (K, 1)).astype(np.int16)
    return out


def stem_class_pairs(manifest_csv: Path) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(manifest_csv.open()))
    pairs = []
    for r in rows:
        stem = f"{r['youtube_id']}_{int(round(float(r['start_s'])))}"
        pairs.append((stem, r["class"]))
    return pairs


def already_done(npy_path: Path, json_path: Path) -> bool:
    if not npy_path.exists() or not json_path.exists():
        return False
    try:
        arr = np.load(npy_path)
        if arr.shape != (K, TOK_PER_SLOT) or arr.dtype != np.int16:
            return False
        captions = json.loads(json_path.read_text())
        return isinstance(captions, list) and len(captions) == K
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

    out_audio = OUT_ROOT / args.split / "tok_audio@512"
    out_desc = OUT_ROOT / args.split / "scene_desc"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_desc.mkdir(parents=True, exist_ok=True)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; loading EnCodec 24kHz...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)
    # encodec uses int operations not friendly to mps in some torch builds → CPU.
    model = model.to("cpu").eval()

    stats = {"done": 0, "skipped": 0, "no_audio": 0, "shape_err": 0}
    for i, (stem, cls) in enumerate(pairs, 1):
        npy_path = out_audio / f"{stem}.npy"
        json_path = out_desc / f"{stem}.json"
        if already_done(npy_path, json_path):
            stats["skipped"] += 1
            continue
        wav_path = CLIPS / cls / stem / "audio.wav"
        if not wav_path.exists():
            stats["no_audio"] += 1
            continue
        try:
            wav = load_audio_for_encodec(wav_path)
            tokens = encode_to_tokens(model, wav)
        except Exception as e:
            print(f"  shape_err {stem}: {e}")
            stats["shape_err"] += 1
            continue
        np.save(npy_path, tokens)
        json_path.write_text(json.dumps([cls] * K))
        stats["done"] += 1
        if i % 50 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] {stats}")
    print(f"\nfinal: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
