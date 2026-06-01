"""Shared helpers for the v4 dataset layout.

Layout:
    raw_root/{class_name}/{split}/{stem}/
        ├── audio.wav            (24 kHz mono, ~3.413s = 81920±12 samples)
        ├── full_10s.wav         (IGNORE)
        ├── frame.jpg            (IGNORE)
        ├── k00.jpg .. k09.jpg   (10 keyframes — used for RGB/depth/normal)
        └── scores.txt

Stem naming: "{src}_{ytid}_{start*10:06d}" where src ∈ {as, vgg}.

Classes are full descriptive labels with spaces, e.g. "horse neighing".

Output layout:
    out_root/{split}/{modality}/{stem}.npy or .json
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


# Top-level files inside raw_root we never iterate over.
SKIP_TOP = {"_MANIFEST.txt", "_index.csv", "_summary.json"}


def iter_stems(raw_root: Path, split: str | None = None,
               class_filter: str | None = None
               ) -> Iterator[tuple[str, str, Path]]:
    """Yield (split, class_name, stem_dir) for every stem under raw_root.

    `split` filters to "train" or "test" if given; None yields both.
    `class_filter` filters to one class if given.
    """
    for cls_dir in sorted(raw_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
            continue
        if cls_dir.name in SKIP_TOP:
            continue
        if class_filter and cls_dir.name != class_filter:
            continue
        cls = cls_dir.name
        for sp_dir in sorted(cls_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            sp = sp_dir.name
            if sp not in ("train", "test"):
                continue
            if split and sp != split:
                continue
            for stem_dir in sorted(sp_dir.iterdir()):
                if stem_dir.is_dir():
                    yield sp, cls, stem_dir


def keyframe_paths(stem_dir: Path) -> list[Path]:
    """k00.jpg..k09.jpg, sorted. Returns whatever exists (may be < 10)."""
    return sorted(stem_dir.glob("k??.jpg"))


def audio_path(stem_dir: Path) -> Path:
    return stem_dir / "audio.wav"


def out_path(out_root: Path, split: str, modality: str,
             stem: str, ext: str = ".npy") -> Path:
    p = out_root / split / modality / f"{stem}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_save_npy(path: Path, arr) -> None:
    """Write .npy atomically. We open tmp as a file handle so numpy doesn't
    silently append a second .npy extension."""
    import numpy as np
    tmp = path.with_name(path.name + ".tmp")  # foo.npy -> foo.npy.tmp
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.rename(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)
