# Splits

Builds the **deterministic, clip-level** split (seed 42) stratified by `class × source`. The result
is committed at the repo root as [`../../splits.json`](../../splits.json) — the artifact a grader can
diff against their own rebuild.

- `08_split_train_test.py` — base clip-level stratified split for a single tokenized set.
- `v5_merge_and_split.py` — **the one used for the reported run**: merges the two tokenized shards
  (`tokenized_v4` + `tokenized_data2`), applies the audio codebook-2 `+1024` offset, and writes the
  final `splits.json`.

```bash
python scripts/splits/v5_merge_and_split.py \
    --sources data/tokenized_v4 data/tokenized_data2 \
    --out_root data/tokenized_v5 --report splits.json
```

Result: 7,347 train / 907 val / 938 test (9,192 total). Same seed + same token set → same partition.
