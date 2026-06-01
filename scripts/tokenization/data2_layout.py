"""data2 layout helpers — Hassan's raw dataset re-tokenized end-to-end.

Layout in:
    raw_root/{split}/{class}/{stem}/
        ├── frame.jpg            (single keyframe — used for depth + normal + rgb)
        ├── audio.wav            (3.413s @ 24 kHz mono — used for tok_audio@512)
        └── full_10s.wav         (IGNORE)

Layout out (matches our v4 output for clean merge):
    out_root/{split}/{modality}/as_{stem}.npy        # int32 npy
    out_root/{split}/scene_desc/as_{stem}.json       # canonical class JSON

We prefix every stem with `as_` so the training split regex
(`^(as|vgg)_(.+)_(\\d{6})$`) parses cleanly. data2 is AudioSet-sourced
(YouTube IDs + start-offset format), so `as_` is correct.

We rewrite class names to match the v4 convention: short label → full label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


# Map data2's short class names → v4-style full names so merging just works.
CLASS_CANONICAL = {
    "cat": "cat meowing",
    "pig": "pig oinking",
    "moo": "cow lowing",
}


def canonical_class(short: str) -> str:
    """data2's short class name (e.g. 'cat') → v4 full name ('cat meowing')."""
    return CLASS_CANONICAL.get(short, short)


def prefix_stem(stem: str) -> str:
    """Add `as_` prefix so training-time clip-id regex matches."""
    if stem.startswith(("as_", "vgg_")):
        return stem
    return f"as_{stem}"


def iter_stems_data2(raw_root: Path, split: str | None = None,
                     class_filter: str | None = None
                     ) -> Iterator[tuple[str, str, str, Path]]:
    """Yield (split, canonical_class, prefixed_stem, stem_dir).

    Layout in: raw_root/{split}/{short_class}/{stem}/  (split-before-class).
    """
    for sp_dir in sorted(raw_root.iterdir()):
        if not sp_dir.is_dir():
            continue
        sp = sp_dir.name
        if sp not in ("train", "test"):
            continue
        if split and sp != split:
            continue
        for cls_dir in sorted(sp_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            short_cls = cls_dir.name
            full_cls = canonical_class(short_cls)
            if class_filter and full_cls != class_filter and short_cls != class_filter:
                continue
            for stem_dir in sorted(cls_dir.iterdir()):
                if not stem_dir.is_dir():
                    continue
                yield sp, full_cls, prefix_stem(stem_dir.name), stem_dir


def frame_path(stem_dir: Path) -> Path:
    return stem_dir / "frame.jpg"


def audio_path(stem_dir: Path) -> Path:
    return stem_dir / "audio.wav"
