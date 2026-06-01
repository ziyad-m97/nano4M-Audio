"""Verify the tokenized/ output has the 5 expected files per stem with the
right shapes & dtypes. Prints a summary and exits 0 only if everything checks.

Expected per stem in tokenized/{split}/:
    tok_rgb@196/{stem}.npy       [10, 196] int32
    tok_audio@512/{stem}.npy     [10, 512] int16
    tok_depth@196/{stem}.npy     [10, 196] int32
    tok_normal@196/{stem}.npy    [10, 196] int32
    scene_desc/{stem}.json       JSON list of 10 strings

Usage:
    python 13_verify.py [--split train|test|both]
    python 13_verify.py --tar          # produce tokenized_YYYYMMDD.tar.gz
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "tokenized"

SPEC = [
    ("tok_rgb@196",    (10, 196), np.int32, ".npy"),
    ("tok_audio@512",  (10, 512), np.int16, ".npy"),
    ("tok_depth@196",  (10, 196), np.int32, ".npy"),
    ("tok_normal@196", (10, 196), np.int32, ".npy"),
    ("scene_desc",     None,      None,     ".json"),
]


def check_split(split: str) -> tuple[int, int, dict]:
    sdir = OUT_ROOT / split
    if not sdir.exists():
        print(f"  [{split}] missing")
        return 0, 0, {}
    stems_by_mod = {}
    for mod, _, _, ext in SPEC:
        d = sdir / mod
        if not d.exists():
            stems_by_mod[mod] = set()
            continue
        stems_by_mod[mod] = {p.stem for p in d.glob(f"*{ext}")}
    all_stems = set.union(*stems_by_mod.values()) if stems_by_mod else set()
    complete_stems = set.intersection(*stems_by_mod.values()) if stems_by_mod else set()

    per_mod_missing = {
        mod: sorted(all_stems - stems_by_mod[mod])
        for mod in stems_by_mod
    }

    shape_errs = {mod: [] for mod, _, _, _ in SPEC}
    sample_stems = sorted(complete_stems)[:10]
    for stem in sample_stems:
        for mod, want_shape, want_dtype, ext in SPEC:
            p = sdir / mod / f"{stem}{ext}"
            if ext == ".npy":
                arr = np.load(p)
                if arr.shape != want_shape or arr.dtype != want_dtype:
                    shape_errs[mod].append(
                        f"{stem}: shape={arr.shape} dtype={arr.dtype}")
            else:
                caps = json.loads(p.read_text())
                if not (isinstance(caps, list) and len(caps) == 10):
                    shape_errs[mod].append(f"{stem}: caps={type(caps).__name__}")

    print(f"  [{split}] all_stems={len(all_stems)} complete={len(complete_stems)}")
    for mod in stems_by_mod:
        missing = len(per_mod_missing[mod])
        bad = len(shape_errs[mod])
        flag = "" if missing == 0 and bad == 0 else " ⚠"
        print(f"    {mod:18s} present={len(stems_by_mod[mod]):5d}"
              f" missing={missing:5d} shape_err={bad}{flag}")
        for s in per_mod_missing[mod][:3]:
            print(f"      missing: {s}")
        for s in shape_errs[mod][:3]:
            print(f"      err:     {s}")

    return len(all_stems), len(complete_stems), shape_errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--tar", action="store_true")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    total_complete = 0
    total_all = 0
    errors_found = False
    for s in splits:
        a, c, errs = check_split(s)
        total_all += a
        total_complete += c
        if any(errs.values()):
            errors_found = True

    print(f"\nsummary: {total_complete}/{total_all} stems complete across {splits}")

    if args.tar:
        stamp = datetime.date.today().strftime("%Y%m%d")
        tar_path = ROOT / f"tokenized_{stamp}.tar.gz"
        print(f"creating {tar_path} ...")
        subprocess.check_call(
            ["tar", "czf", str(tar_path), "-C", str(ROOT), "tokenized"])
        print(f"wrote {tar_path}")

    return 1 if errors_found else 0


if __name__ == "__main__":
    raise SystemExit(main())
