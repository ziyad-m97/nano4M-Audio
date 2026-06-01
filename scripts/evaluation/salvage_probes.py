"""Salvage probes — 3 quick tests to see if there's more cross-modal signal
than the naive audio-only / argmax / top-1 setup captured.

1. Top-K class accuracy (class-logit ranking instead of decode-then-match)
2. Multi-modal input → scene_desc (audio + rgb + depth + normal)
3. Top-K accuracy with temperature sampling (for diversity)

Output: eval_out/eval_results/salvage_probes.json + per-class breakdowns.
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

NANO4M_DIR = "/path/to/nanofm"
DATA_ROOT  = "/path/to/nano4M-Audio/data/tokenized_v5"
CFG_PATH   = f"{NANO4M_DIR}/cfgs/nano4M/animal_full_5mod_v5.yaml"
CKPT_PATH  = "/path/to/nano4M-Audio/runs/animal_full_5mod_v5_fresh/checkpoint-final.safetensors"
OUT = Path("eval_out/eval_results"); OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, NANO4M_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42); torch.manual_seed(42)

CLASSES_CANONICAL = [
    "cat meowing", "chicken clucking", "cow lowing", "coyote howling",
    "dog barking", "duck quacking", "horse neighing", "lions roaring",
    "pig oinking", "sheep bleating", "pigeon cooing",
]
CLASSES_SHORT = [c.split()[0] for c in CLASSES_CANONICAL]
N_CLASSES = len(CLASSES_CANONICAL)

cfg = OmegaConf.load(CFG_PATH)
MODALITIES   = list(cfg.global_vars.modalities)  # rgb, audio, depth, normal, scene_desc
MAX_SEQ_LENS = list(cfg.global_vars.max_seq_lens)
MOD_TO_IDX   = {m: i for i, m in enumerate(MODALITIES)}
SD_VOCAB = cfg.global_vars.vocab_sizes[MOD_TO_IDX["scene_desc"]]  # 50304

from nanofm.models.fourm import FourM
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset

print("Loading model...")
model = instantiate(cfg.model_config).to(DEVICE).eval()
from safetensors.torch import load_file
state = load_file(CKPT_PATH, device="cpu")
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# GPT-2 tokenizer for decoding class-key tokens.
# Training used TemplateProcessing("[SOS] $A [EOS]") so during training the
# scene_desc target sequence was [SOS_id, " cat"_id, " meowing"_id, EOS_id, PAD, ...]
# At eval, we only need the ID of " cat" (the first BPE piece of the class) —
# that's what the model should predict at scene_desc position 1.
from transformers import GPT2Tokenizer
gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
print(f"GPT-2 tokenizer vocab: {len(gpt2)}")

def first_class_token(short_class_name: str) -> int:
    """Token ID for the first BPE piece of ' {class_name}' — what the model
    should predict at scene_desc position 1 (first content after [SOS])."""
    ids = gpt2.encode(" " + short_class_name, add_special_tokens=False)
    return ids[0]

CLASS_KEY_TOKENS = {c: first_class_token(s)
                    for c, s in zip(CLASSES_CANONICAL, CLASSES_SHORT)}
print("Class key tokens:")
for c, t in CLASS_KEY_TOKENS.items():
    print(f"  {c:24s} token_id={t} decoded={gpt2.decode([t])!r}")

# Dataset (no transforms)
test_ds = SimpleMultimodalDataset(
    root_dir=DATA_ROOT, split="test", modalities=MODALITIES,
    transforms=None, sample_from_k_augmentations=10,
    text_tokenizer_path="gpt2", text_max_length=64,
)
print(f"Test set: {len(test_ds)}")


def read_class(stem: str) -> str:
    return json.loads((Path(DATA_ROOT) / "test" / "scene_desc" / f"{stem}.json").read_text())[0]


def get_sample(idx: int):
    fn = test_ds.file_names[idx]
    aug = np.random.randint(0, test_ds.sample_from_k_augmentations)
    out = {"__stem__": fn, "__class__": read_class(fn)}
    for mod in MODALITIES:
        ext  = test_ds.modality_extensions[mod]
        path = Path(test_ds.root_dir) / test_ds.split / mod / f"{fn}{ext}"
        if "tok" in mod:
            arr = np.load(path)
            out[mod] = torch.from_numpy(arr[aug]).long()
        else:
            captions = json.loads(path.read_text())
            tokenized = test_ds.text_tokenizer(
                captions[aug], max_length=test_ds.text_max_length,
                padding="max_length", truncation=True, return_tensors="pt")
            out[mod] = tokenized["input_ids"][0].long()
    return out


def build_batch(input_mods: dict, target_mods: dict):
    enc_t, enc_m, enc_p = [], [], []
    for m, (toks, pos) in input_mods.items():
        p = torch.as_tensor(pos, dtype=torch.long, device=DEVICE)
        t = torch.as_tensor(toks, dtype=torch.long, device=DEVICE)
        enc_t.append(t); enc_p.append(p)
        enc_m.append(torch.full_like(p, MOD_TO_IDX[m]))
    dec_m, dec_p = [], []
    for m, pos in target_mods.items():
        p = torch.as_tensor(pos, dtype=torch.long, device=DEVICE)
        dec_p.append(p)
        dec_m.append(torch.full_like(p, MOD_TO_IDX[m]))
    return dict(
        enc_input_tokens=torch.cat(enc_t).unsqueeze(0),
        enc_input_modalities=torch.cat(enc_m).unsqueeze(0),
        enc_input_positions=torch.cat(enc_p).unsqueeze(0),
        dec_input_modalities=torch.cat(dec_m).unsqueeze(0) if dec_m else
                             torch.zeros(1, 0, dtype=torch.long, device=DEVICE),
        dec_input_positions=torch.cat(dec_p).unsqueeze(0) if dec_p else
                            torch.zeros(1, 0, dtype=torch.long, device=DEVICE),
        enc_pad_mask=torch.ones(1, sum(p.shape[0] for p in enc_p), dtype=torch.bool, device=DEVICE),
        dec_pad_mask=torch.ones(1, sum(p.shape[0] for p in dec_p) if dec_p else 0, dtype=torch.bool, device=DEVICE),
    )


@torch.no_grad()
def predict_class_logits(input_mods: dict) -> dict:
    """Return {class: log P(class_key_token at position 1 | input_mods)} for all 11 classes.
    Use position 1 (first content token after SOS) — that's where the class name lives."""
    args = build_batch(input_mods, {"scene_desc": [1]})  # only position 1
    logits = model.forward_model(**args)[0, 0]  # [MAX_VOCAB]
    logp = F.log_softmax(logits.float(), dim=-1)
    return {c: logp[t].item() for c, t in CLASS_KEY_TOKENS.items()}


def topk_accuracy(predictions_logp: dict, true_class: str, k: int) -> bool:
    """predictions_logp: {class: log P}. True class in top-K?"""
    ranked = sorted(predictions_logp.items(), key=lambda x: -x[1])
    return true_class in [c for c, _ in ranked[:k]]


# ============================================================
# TEST 1: Top-K class accuracy (audio-only input)
# ============================================================
print("\n" + "="*60)
print("TEST 1: Audio-only → class logit ranking, Top-K")
print("="*60)
AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]

hits = {1: 0, 3: 0, 5: 0}
per_class_hits = defaultdict(lambda: {1: 0, 3: 0, 5: 0, "total": 0})
N = len(test_ds)
for idx in tqdm(range(N), desc="audio-only top-K"):
    s = get_sample(idx)
    true_c = s["__class__"]
    if true_c not in CLASSES_CANONICAL:
        continue
    logp = predict_class_logits(
        {"tok_audio@512": (s["tok_audio@512"], list(range(AUDIO_LEN)))})
    per_class_hits[true_c]["total"] += 1
    for k in (1, 3, 5):
        if topk_accuracy(logp, true_c, k):
            hits[k] += 1
            per_class_hits[true_c][k] += 1

print(f"\nTop-1 = {hits[1]/N:.1%}  (random {1/N_CLASSES:.1%})")
print(f"Top-3 = {hits[3]/N:.1%}  (random {3/N_CLASSES:.1%})")
print(f"Top-5 = {hits[5]/N:.1%}  (random {5/N_CLASSES:.1%})")
print("\nPer-class (Top-1 / Top-3 / Top-5):")
for c in CLASSES_CANONICAL:
    p = per_class_hits[c]
    t = max(p["total"], 1)
    print(f"  {c:24s} {p[1]/t:5.1%} / {p[3]/t:5.1%} / {p[5]/t:5.1%}  (n={p['total']})")

test1 = {
    "top1": hits[1]/N, "top3": hits[3]/N, "top5": hits[5]/N,
    "per_class": {c: {**dict(per_class_hits[c]), "ratios": {
        k: per_class_hits[c][k] / max(per_class_hits[c]["total"], 1)
        for k in (1, 3, 5)}} for c in CLASSES_CANONICAL},
}


# ============================================================
# TEST 2: Multi-modal input (audio + rgb + depth + normal) → scene_desc
# ============================================================
print("\n" + "="*60)
print("TEST 2: Multi-modal input (audio+rgb+depth+normal) → class")
print("="*60)
RGB_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["tok_rgb@196"]]
DEPTH_LEN  = MAX_SEQ_LENS[MOD_TO_IDX["tok_depth@196"]]
NORMAL_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_normal@196"]]

hits = {1: 0, 3: 0, 5: 0}
per_class_hits = defaultdict(lambda: {1: 0, 3: 0, 5: 0, "total": 0})
for idx in tqdm(range(N), desc="multi-mod top-K"):
    s = get_sample(idx)
    true_c = s["__class__"]
    if true_c not in CLASSES_CANONICAL:
        continue
    logp = predict_class_logits({
        "tok_audio@512":  (s["tok_audio@512"],  list(range(AUDIO_LEN))),
        "tok_rgb@196":    (s["tok_rgb@196"],    list(range(RGB_LEN))),
        "tok_depth@196":  (s["tok_depth@196"],  list(range(DEPTH_LEN))),
        "tok_normal@196": (s["tok_normal@196"], list(range(NORMAL_LEN))),
    })
    per_class_hits[true_c]["total"] += 1
    for k in (1, 3, 5):
        if topk_accuracy(logp, true_c, k):
            hits[k] += 1
            per_class_hits[true_c][k] += 1

print(f"\nTop-1 = {hits[1]/N:.1%}")
print(f"Top-3 = {hits[3]/N:.1%}")
print(f"Top-5 = {hits[5]/N:.1%}")
print("\nPer-class (Top-1 / Top-3 / Top-5):")
for c in CLASSES_CANONICAL:
    p = per_class_hits[c]
    t = max(p["total"], 1)
    print(f"  {c:24s} {p[1]/t:5.1%} / {p[3]/t:5.1%} / {p[5]/t:5.1%}  (n={p['total']})")

test2 = {
    "top1": hits[1]/N, "top3": hits[3]/N, "top5": hits[5]/N,
    "per_class": {c: {**dict(per_class_hits[c]), "ratios": {
        k: per_class_hits[c][k] / max(per_class_hits[c]["total"], 1)
        for k in (1, 3, 5)}} for c in CLASSES_CANONICAL},
}


# ============================================================
# TEST 3: RGB-only → scene_desc (control: how much does VISION encode class?)
# ============================================================
print("\n" + "="*60)
print("TEST 3: RGB-only → class (control)")
print("="*60)
hits = {1: 0, 3: 0, 5: 0}
per_class_hits = defaultdict(lambda: {1: 0, 3: 0, 5: 0, "total": 0})
for idx in tqdm(range(N), desc="rgb-only top-K"):
    s = get_sample(idx)
    true_c = s["__class__"]
    if true_c not in CLASSES_CANONICAL:
        continue
    logp = predict_class_logits({
        "tok_rgb@196": (s["tok_rgb@196"], list(range(RGB_LEN)))})
    per_class_hits[true_c]["total"] += 1
    for k in (1, 3, 5):
        if topk_accuracy(logp, true_c, k):
            hits[k] += 1
            per_class_hits[true_c][k] += 1

print(f"\nTop-1 = {hits[1]/N:.1%}")
print(f"Top-3 = {hits[3]/N:.1%}")
print(f"Top-5 = {hits[5]/N:.1%}")
print("\nPer-class (Top-1 / Top-3 / Top-5):")
for c in CLASSES_CANONICAL:
    p = per_class_hits[c]
    t = max(p["total"], 1)
    print(f"  {c:24s} {p[1]/t:5.1%} / {p[3]/t:5.1%} / {p[5]/t:5.1%}  (n={p['total']})")

test3 = {
    "top1": hits[1]/N, "top3": hits[3]/N, "top5": hits[5]/N,
    "per_class": {c: {**dict(per_class_hits[c]), "ratios": {
        k: per_class_hits[c][k] / max(per_class_hits[c]["total"], 1)
        for k in (1, 3, 5)}} for c in CLASSES_CANONICAL},
}


# ============================================================
# Save + summarize
# ============================================================
out = {
    "n_test_samples": N,
    "n_classes": N_CLASSES,
    "random_baselines": {"top1": 1/N_CLASSES, "top3": 3/N_CLASSES, "top5": 5/N_CLASSES},
    "test1_audio_only":    test1,
    "test2_multimodal":    test2,
    "test3_rgb_only":      test3,
}
(OUT / "salvage_probes.json").write_text(json.dumps(out, indent=2))

print("\n" + "="*60)
print("SALVAGE SUMMARY (Top-1 / Top-3 / Top-5)")
print("="*60)
print(f"{'Input':<24s} {'Top-1':>8s} {'Top-3':>8s} {'Top-5':>8s}")
print(f"{'random':<24s} {1/N_CLASSES:>7.1%} {3/N_CLASSES:>7.1%} {5/N_CLASSES:>7.1%}")
print(f"{'audio-only':<24s} {test1['top1']:>7.1%} {test1['top3']:>7.1%} {test1['top5']:>7.1%}")
print(f"{'multi-modal':<24s} {test2['top1']:>7.1%} {test2['top3']:>7.1%} {test2['top5']:>7.1%}")
print(f"{'rgb-only (control)':<24s} {test3['top1']:>7.1%} {test3['top3']:>7.1%} {test3['top5']:>7.1%}")
print(f"\nSaved: {OUT}/salvage_probes.json")
