"""Merge v4 + data2 tokenized outputs into a single, clip-level-stratified
80/10/10 train/val/test dataset, applying audio codebook-2 offset on the fly.

Inputs (already on SCITAS scratch):
    /path/to/nano4M-Audio/data/tokenized_v4/{train|test}/{mod}/{stem}.npy
    /path/to/nano4M-Audio/data/tokenized_data2/{train|test}/{mod}/{stem}.npy

Output:
    /path/to/nano4M-Audio/data/tokenized_v5/{train|val|test}/{mod}/{stem}.npy

Key transforms applied at write time:
- Audio: codebook-2 (odd positions) += 1024 → vocab effective 2048 (`overlap_vocab=False`
  per-modality, so ids in [0, 2047] cleanly distinguish C1 from C2).
- Splits: clip-level (regex extract clip_id), stratified per (class × source).
  Original {train,test} folders are ignored; we re-split from scratch.

Stem regex:
    ^(as|vgg)_(.+)_(\\d{6})$   — captures src, ytid, start_offset.
clip_id := "{src}_{ytid}"  → all stems from same clip stay in the same split.

Uses np.save + atomic rename; symlinks for non-audio modalities (no offset
needed), real .npy writes only for audio. JSON for scene_desc is copied
verbatim.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SEED = 42
PROPORTIONS = (0.80, 0.10, 0.10)
MIN_PER_CLASS_PER_SPLIT = 10  # warn if a class has < this in any split

STEM_RE = re.compile(r"^(as|vgg)_(.+)_(\d{6})$")

MODS_NPY = [
    ("tok_rgb@196",    16384),
    ("tok_depth@196",  8192),
    ("tok_normal@196", 8192),
    ("tok_audio@512",  2048),   # post-offset vocab
]
MOD_JSON = "scene_desc"


def parse_stem(stem: str) -> tuple[str, str]:
    """(clip_id, src). Raises if unparseable."""
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"unparseable stem: {stem!r}")
    src, ytid, _ = m.groups()
    return f"{src}_{ytid}", src


def stem_class(stem: str, scene_desc_root: Path) -> str:
    """Read scene_desc/{stem}.json to get the class name (first entry)."""
    p = scene_desc_root / f"{stem}.json"
    return json.loads(p.read_text())[0]


def gather_stems(src_root: Path) -> dict[str, dict[str, Path]]:
    """For src_root/{train|test}/{mod}/{stem}.npy, return
       {stem: {mod: path, ...}}. Only stems with ALL 5 mods are kept."""
    stems: dict[str, dict[str, Path]] = defaultdict(dict)
    for split_dir in src_root.iterdir():
        if not split_dir.is_dir() or split_dir.name not in ("train", "test"):
            continue
        for mod_name, _ in MODS_NPY:
            mod_dir = split_dir / mod_name
            if not mod_dir.exists():
                continue
            for p in mod_dir.glob("*.npy"):
                stems[p.stem][mod_name] = p
        sd = split_dir / MOD_JSON
        if sd.exists():
            for p in sd.glob("*.json"):
                stems[p.stem][MOD_JSON] = p
    # Keep only stems with all 5 modalities present.
    keep = {}
    needed = {m for m, _ in MODS_NPY} | {MOD_JSON}
    for stem, mods in stems.items():
        if set(mods.keys()) >= needed:
            keep[stem] = mods
    return keep


def apply_audio_offset_save(src_npy: Path, dst_npy: Path) -> None:
    """Load audio .npy, offset codebook-2 (odd positions) by +1024, save."""
    arr = np.load(src_npy)
    # Interleaved layout: C1[0], C2[0], C1[1], C2[1], ...
    # Even indices = C1 (stay [0, 1023]); odd indices = C2 (+= 1024).
    out = arr.copy()
    out[:, 1::2] += 1024
    assert out.min() >= 0 and out.max() < 2048, \
        f"{src_npy}: out of [0,2047], min={out.min()} max={out.max()}"
    tmp = dst_npy.with_name(dst_npy.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, out.astype(np.int32))
    tmp.rename(dst_npy)


def symlink_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def split_clips(stems: list[str], scene_desc_lookup: dict[str, str],
                rng: random.Random) -> dict[str, set[str]]:
    """Clip-level stratified by (class × source). Returns dict split→set(stem)."""
    clip_to_stems: dict[str, list[str]] = defaultdict(list)
    clip_stratum: dict[str, str] = {}
    for stem in stems:
        cid, src = parse_stem(stem)
        cls = scene_desc_lookup[stem]
        clip_to_stems[cid].append(stem)
        clip_stratum[cid] = f"{cls}__{src}"

    stratum_clips: dict[str, list[str]] = defaultdict(list)
    for cid, strat in clip_stratum.items():
        stratum_clips[strat].append(cid)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for strat, clips in stratum_clips.items():
        clips = sorted(clips)
        rng.shuffle(clips)
        n = len(clips)
        n_train = int(PROPORTIONS[0] * n)
        n_val = int(PROPORTIONS[1] * n)
        for split_name, lst in [
            ("train", clips[:n_train]),
            ("val", clips[n_train:n_train + n_val]),
            ("test", clips[n_train + n_val:]),
        ]:
            for cid in lst:
                splits[split_name].extend(clip_to_stems[cid])

    out = {k: set(v) for k, v in splits.items()}
    # Guard rails
    assert out["train"].isdisjoint(out["val"])
    assert out["train"].isdisjoint(out["test"])
    assert out["val"].isdisjoint(out["test"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, type=Path,
                    help="One or more tokenized roots (v4, data2, ...)")
    ap.add_argument("--out_root", required=True, type=Path,
                    help="Final merged tokenized_v5 root (will hold train/val/test)")
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="Symlink non-audio files instead of copy (default)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path to write splits.json + per-class counts")
    args = ap.parse_args()

    rng = random.Random(SEED)

    # 1) Gather stems from all sources, dedup by stem name (data2 has as_ prefix
    #    so it won't collide with v4 stems unless someone re-named identically).
    all_stems: dict[str, dict[str, Path]] = {}
    per_source: dict[str, int] = {}
    for src in args.sources:
        s = gather_stems(src)
        per_source[str(src)] = len(s)
        for stem, mods in s.items():
            if stem in all_stems:
                print(f"WARN duplicate stem {stem!r} (skipping, keeping first)",
                      file=sys.stderr)
                continue
            all_stems[stem] = mods
    print(f"gathered {len(all_stems)} unique stems from {len(args.sources)} sources")
    for k, v in per_source.items():
        print(f"  {k}: {v}")

    # 2) Build (stem → class) lookup from scene_desc.
    scene_desc_lookup: dict[str, str] = {}
    bad_stems = []
    for stem, mods in all_stems.items():
        try:
            scene_desc_lookup[stem] = json.loads(mods[MOD_JSON].read_text())[0]
        except Exception as e:
            bad_stems.append((stem, str(e)))
    for stem, err in bad_stems[:5]:
        print(f"  WARN bad scene_desc {stem}: {err}", file=sys.stderr)

    # 3) Drop stems that fail the regex (shouldn't happen after our prefixing).
    valid_stems = []
    for stem in scene_desc_lookup.keys():
        try:
            parse_stem(stem)
            valid_stems.append(stem)
        except ValueError as e:
            print(f"  WARN dropping {stem}: {e}", file=sys.stderr)
    print(f"valid stems after regex: {len(valid_stems)}")

    # 4) Stratified clip-level split.
    splits = split_clips(valid_stems, scene_desc_lookup, rng)
    for sp, ss in splits.items():
        cls_counts = Counter(scene_desc_lookup[s] for s in ss)
        print(f"\n[{sp}] n={len(ss)} classes={len(cls_counts)}")
        for cls, n in sorted(cls_counts.items()):
            flag = " ⚠ low" if n < MIN_PER_CLASS_PER_SPLIT else ""
            print(f"  {cls:24s} {n}{flag}")

    # 5) Write out_root/{split}/{mod}/{stem}.npy or .json
    for sp, ss in splits.items():
        for mod_name, _ in MODS_NPY:
            (args.out_root / sp / mod_name).mkdir(parents=True, exist_ok=True)
        (args.out_root / sp / MOD_JSON).mkdir(parents=True, exist_ok=True)
        for stem in sorted(ss):
            mods = all_stems[stem]
            for mod_name, _ in MODS_NPY:
                src_p = mods[mod_name]
                dst_p = args.out_root / sp / mod_name / f"{stem}.npy"
                if dst_p.exists() or dst_p.is_symlink():
                    continue
                if mod_name == "tok_audio@512":
                    apply_audio_offset_save(src_p, dst_p)
                else:
                    symlink_or_copy(src_p, dst_p, args.symlink)
            # scene_desc: just symlink the .json (no transform)
            sd_src = mods[MOD_JSON]
            sd_dst = args.out_root / sp / MOD_JSON / f"{stem}.json"
            if not (sd_dst.exists() or sd_dst.is_symlink()):
                symlink_or_copy(sd_src, sd_dst, args.symlink)

    if args.report:
        report = {
            "seed": SEED,
            "proportions": PROPORTIONS,
            "stratification": "class_x_source",
            "min_per_class_per_split": MIN_PER_CLASS_PER_SPLIT,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "splits": {k: sorted(v) for k, v in splits.items()},
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.report}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
