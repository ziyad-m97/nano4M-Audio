"""Split final_manifest.csv into train (90%) / test (10%) by stem.

Stem = "{youtube_id}_{int(round(start_s))}"  — matches clip dir naming.

Output:
    metadata/manifest_train.csv
    metadata/manifest_test.csv

Deterministic via --seed.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
SRC = META / "final_manifest.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", action="store_true", default=True,
                    help="Stratify the split per class (default on).")
    args = ap.parse_args()

    rows = list(csv.DictReader(SRC.open()))
    for r in rows:
        r["stem"] = f"{r['youtube_id']}_{int(round(float(r['start_s'])))}"

    rng = random.Random(args.seed)
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)

    train, test = [], []
    for cls, group in by_class.items():
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * args.test_frac)))
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)

    fields = list(rows[0].keys())
    for split, data in [("train", train), ("test", test)]:
        out = META / f"manifest_{split}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print(f"wrote {out}: {len(data)}")
        cls_counts: dict[str, int] = {}
        for r in data:
            cls_counts[r["class"]] = cls_counts.get(r["class"], 0) + 1
        for cls, n in sorted(cls_counts.items()):
            print(f"  {cls:8s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
