"""Verify the v4 tokenized output: shapes/dtypes/vocab bounds + per-class counts.

Walks out_root/{train,test}/{modality}/. Cross-checks against raw_root stem list.
Prints a missing-files report and per-class counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_layout import iter_stems

SPEC = [
    ("tok_rgb@196",    (10, 196), np.int32, ".npy", 16384),
    ("tok_depth@196",  (10, 196), np.int32, ".npy", 8192),
    ("tok_normal@196", (10, 196), np.int32, ".npy", 8192),
    ("tok_audio@512",  (10, 512), np.int32, ".npy", 1024),
    ("scene_desc",     None,      None,     ".json", None),
]


def check_split(out_root: Path, raw_root: Path, split: str) -> dict:
    raw_stems_by_class: dict[str, set[str]] = {}
    for sp, cls, sd in iter_stems(raw_root, split):
        raw_stems_by_class.setdefault(cls, set()).add(sd.name)
    raw_stems = set().union(*raw_stems_by_class.values()) if raw_stems_by_class else set()

    print(f"\n[{split}] raw stems: {len(raw_stems)}  classes: {len(raw_stems_by_class)}")
    for cls, s in sorted(raw_stems_by_class.items()):
        print(f"  {cls:24s} {len(s)}")

    present_by_mod: dict[str, set[str]] = {}
    shape_errs: dict[str, list[str]] = {m: [] for m, *_ in SPEC}
    vocab_errs: dict[str, list[str]] = {m: [] for m, *_ in SPEC}
    for mod, want_shape, want_dtype, ext, vocab in SPEC:
        d = out_root / split / mod
        if not d.exists():
            present_by_mod[mod] = set()
            continue
        present_by_mod[mod] = {p.stem for p in d.glob(f"*{ext}")}
        # sample-check first 50 files for shape + vocab
        for p in sorted(d.iterdir())[:50]:
            try:
                if ext == ".npy":
                    arr = np.load(p)
                    if arr.shape != want_shape or arr.dtype != want_dtype:
                        shape_errs[mod].append(
                            f"{p.name}: shape={arr.shape} dtype={arr.dtype}")
                    elif vocab is not None and (arr.min() < 0 or arr.max() >= vocab):
                        vocab_errs[mod].append(
                            f"{p.name}: min={arr.min()} max={arr.max()} vocab={vocab}")
                else:
                    caps = json.loads(p.read_text())
                    if not (isinstance(caps, list) and len(caps) == 10):
                        shape_errs[mod].append(f"{p.name}: caps={type(caps).__name__}")
            except Exception as e:
                shape_errs[mod].append(f"{p.name}: load_err {type(e).__name__}")

    print(f"\n[{split}] per-modality:")
    for mod, *_ in SPEC:
        n = len(present_by_mod.get(mod, set()))
        missing = raw_stems - present_by_mod.get(mod, set())
        nshape = len(shape_errs[mod])
        nvocab = len(vocab_errs[mod])
        flag = "" if not missing and not nshape and not nvocab else " ⚠"
        print(f"  {mod:18s} present={n:5d}  missing={len(missing):5d}"
              f"  shape_err={nshape}  vocab_err={nvocab}{flag}")
        for s in sorted(missing)[:3]:
            print(f"    missing: {s}")
        for s in shape_errs[mod][:2]:
            print(f"    shape:   {s}")
        for s in vocab_errs[mod][:2]:
            print(f"    vocab:   {s}")
    return present_by_mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    args = ap.parse_args()
    for sp in ("train", "test"):
        check_split(args.out_root, args.raw_root, sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
