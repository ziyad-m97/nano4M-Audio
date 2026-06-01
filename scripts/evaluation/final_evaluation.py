"""nano4M-Audio — Final inference & evaluation notebook (jupytext-style .py).

8 cells matching `inference_debugging_notebook.md`, adapted to OUR layout:
- CFG : /path/to/nanofm/cfgs/nano4M/animal_full_5mod_v5.yaml
- CKPT: /path/to/nano4M-Audio/runs/animal_full_5mod_v5_fresh/checkpoint-final.safetensors
        (falls back to /scratch/.../animal_full_5mod_v5/checkpoint-final.safetensors if absent)
- DATA: /path/to/nano4M-Audio/data/tokenized_v5
- SPLITS: /path/to/nano4M-Audio/data/tokenized_v5/splits.json

Class label convention:
- Our scene_desc on disk uses canonical full names ("cat meowing", "horse neighing", ...)
- The pitch text in the prompt uses short names ("cat", "horse", ...)
- We map both ways internally; classification compares against the FIRST WORD
  of the canonical label.

Each cell is wrapped in try/except so a failure in one doesn't kill the rest.
"""
# %% cell-1 setup --------------------------------------------------------------
import os, sys, json, traceback, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from tqdm import tqdm

NANO4M_DIR = "/path/to/nanofm"
DATA_ROOT  = "/path/to/nano4M-Audio/data/tokenized_v5"
SPLITS_JSON = f"{DATA_ROOT}/splits.json"
CFG_PATH   = f"{NANO4M_DIR}/cfgs/nano4M/animal_full_5mod_v5.yaml"

CKPT_CANDIDATES = [
    "/path/to/nano4M-Audio/runs/animal_full_5mod_v5_fresh/checkpoint-final.safetensors",
    "/path/to/nano4M-Audio/runs/animal_full_5mod_v5_fresh/checkpoint-final.pth",
    "/path/to/nano4M-Audio/runs/animal_full_5mod_v5/checkpoint-final.safetensors",
    "/path/to/nano4M-Audio/runs/animal_full_5mod_v5/checkpoint-final.pth",
]
CKPT_PATH = next((p for p in CKPT_CANDIDATES if Path(p).exists()), None)

FIG_DIR     = Path("eval_out/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("eval_out/eval_results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Path so we can import nanofm
sys.path.insert(0, NANO4M_DIR)

np.random.seed(42); torch.manual_seed(42); random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical class labels (the strings stored in scene_desc/*.json).
# Order MATTERS — this becomes the class index used in confusion matrices.
CLASSES_CANONICAL = [
    "cat meowing",     "chicken clucking", "cow lowing",   "coyote howling",
    "dog barking",     "duck quacking",    "horse neighing","lions roaring",
    "pig oinking",     "sheep bleating",   "pigeon cooing",
]
CLASSES_SHORT = [c.split()[0] for c in CLASSES_CANONICAL]  # ["cat","chicken",...]
N_CLASSES = len(CLASSES_CANONICAL)

print(f"Device: {DEVICE}")
print(f"Classes ({N_CLASSES}): {CLASSES_SHORT}")
print(f"Checkpoint: {CKPT_PATH}")
assert CKPT_PATH is not None, "no checkpoint found"

# --- Load cfg + model ----------------------------------------------------------
cfg = OmegaConf.load(CFG_PATH)
MODALITIES   = list(cfg.global_vars.modalities)
VOCAB_SIZES  = list(cfg.global_vars.vocab_sizes)
MAX_SEQ_LENS = list(cfg.global_vars.max_seq_lens)
MAX_VOCAB    = max(VOCAB_SIZES)        # 50304 — the unified embedding table size
MOD_TO_IDX   = {m: i for i, m in enumerate(MODALITIES)}

print(f"Modalities: {MODALITIES}")
print(f"vocab sizes: {VOCAB_SIZES}  (max={MAX_VOCAB})")
print(f"max_seq_lens: {MAX_SEQ_LENS}")

from nanofm.models.fourm import FourM
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset

model = instantiate(cfg.model_config).to(DEVICE).eval()
print(f"Loading state dict from {CKPT_PATH} ...")
if CKPT_PATH.endswith(".safetensors"):
    from safetensors.torch import load_file
    state = load_file(CKPT_PATH, device="cpu")
else:
    state = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
# Strip any DDP prefix
state = {k.replace("module.", ""): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}")
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  params: {n_params:.1f}M")

# --- Load dataset (no masking transform — we'll do inference batch building) --
test_ds  = SimpleMultimodalDataset(
    root_dir=DATA_ROOT, split="test",  modalities=MODALITIES,
    transforms=None, sample_from_k_augmentations=10, text_tokenizer_path="gpt2",
    text_max_length=64,
)
train_ds = SimpleMultimodalDataset(
    root_dir=DATA_ROOT, split="train", modalities=MODALITIES,
    transforms=None, sample_from_k_augmentations=10, text_tokenizer_path="gpt2",
    text_max_length=64,
)

print(f"\nTest set : {len(test_ds)} stems")
print(f"Train set: {len(train_ds)} stems")

# Helper: read the class for a stem (peek scene_desc JSON directly — bypass dataset's BPE).
def read_class_canonical(split: str, stem: str) -> str:
    p = Path(DATA_ROOT) / split / "scene_desc" / f"{stem}.json"
    return json.loads(p.read_text())[0]

# --- Load detokenizers for RGB/Depth/Normal + EnCodec for audio + GPT-2 -------
try:
    from fourm.vq.vqvae import DiVAE
    rgb_detok    = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").to(DEVICE).eval()
    depth_detok  = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(DEVICE).eval()
    normal_detok = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_normal_8k_224-448").to(DEVICE).eval()
    DETOK_OK = True
except Exception as e:
    print(f"DETOK load failed: {e}")
    DETOK_OK = False

try:
    from encodec import EncodecModel
    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(1.5)
    encodec = encodec.to(DEVICE).eval()
    ENC_OK = True
except Exception as e:
    print(f"EnCodec load failed: {e}")
    ENC_OK = False

from transformers import GPT2Tokenizer
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

print(f"\ndetokenizers: DiVAE={DETOK_OK}  EnCodec={ENC_OK}  GPT2 tokenizer=OK")


# %% helpers -------------------------------------------------------------------
def get_modality_tokens(ds: SimpleMultimodalDataset, idx: int) -> dict:
    """Return {modality: 1D LongTensor of tokens} for one stem.
    Bypasses the masking transform; uses the SAME K-aug as dataset would draw.
    """
    file_name = ds.file_names[idx]
    aug = np.random.randint(0, ds.sample_from_k_augmentations)
    out = {}
    for mod in MODALITIES:
        ext  = ds.modality_extensions[mod]
        path = Path(ds.root_dir) / ds.split / mod / f"{file_name}{ext}"
        if "tok" in mod:
            arr = np.load(path)
            out[mod] = torch.from_numpy(arr[aug]).long()
        else:  # scene_desc — string list
            captions = json.loads(path.read_text())
            caption = captions[aug]
            tokenized = ds.text_tokenizer(
                caption, max_length=ds.text_max_length, padding="max_length",
                truncation=True, return_tensors="pt")
            out[mod] = tokenized["input_ids"][0].long()
    out["__stem__"] = file_name
    out["__class__"] = read_class_canonical(ds.split, file_name)
    return out


def build_eval_batch(input_mods: dict, target_mods: dict):
    """Pack (input_mods, target_mods) into the tensors `forward_model` wants.

    input_mods: {modality_name: (1D LongTensor of tokens, 1D Long list of positions)}
    target_mods: {modality_name: 1D Long list of positions to predict}

    Returns: dict of args ready for model.forward_model(**args), all on DEVICE.
    """
    enc_tokens, enc_mods, enc_pos = [], [], []
    for m, (toks, positions) in input_mods.items():
        # Move each chunk to DEVICE before cat to avoid CPU/CUDA mix from
        # mixed sources (sample tensors are CPU, maskgit-built tokens are CUDA).
        positions = torch.as_tensor(positions, dtype=torch.long, device=DEVICE)
        toks = torch.as_tensor(toks, dtype=torch.long, device=DEVICE)
        assert positions.shape[0] == toks.shape[0], (
            f"{m}: {positions.shape=} vs {toks.shape=}")
        enc_tokens.append(toks)
        enc_pos.append(positions)
        enc_mods.append(torch.full_like(positions, MOD_TO_IDX[m]))

    dec_mods, dec_pos = [], []
    for m, positions in target_mods.items():
        positions = torch.as_tensor(positions, dtype=torch.long, device=DEVICE)
        dec_pos.append(positions)
        dec_mods.append(torch.full_like(positions, MOD_TO_IDX[m]))

    enc_tokens = torch.cat(enc_tokens).unsqueeze(0)  # [1, N], already on DEVICE
    enc_mods   = torch.cat(enc_mods).unsqueeze(0)
    enc_pos    = torch.cat(enc_pos).unsqueeze(0)
    dec_mods   = (torch.cat(dec_mods).unsqueeze(0)
                  if dec_mods else torch.zeros(1, 0, dtype=torch.long, device=DEVICE))
    dec_pos    = (torch.cat(dec_pos).unsqueeze(0)
                  if dec_pos else torch.zeros(1, 0, dtype=torch.long, device=DEVICE))
    enc_pad = torch.ones_like(enc_tokens, dtype=torch.bool)
    dec_pad = torch.ones_like(dec_mods, dtype=torch.bool)
    return dict(
        enc_input_tokens=enc_tokens,
        enc_input_modalities=enc_mods,
        enc_input_positions=enc_pos,
        dec_input_modalities=dec_mods,
        dec_input_positions=dec_pos,
        enc_pad_mask=enc_pad,
        dec_pad_mask=dec_pad,
    )


@torch.no_grad()
def forward_logits(input_mods: dict, target_mods: dict) -> dict:
    """Run the model and split logits per target modality.
    Returns {modality_name: LongTensor [n_positions, MAX_VOCAB]}."""
    args = build_eval_batch(input_mods, target_mods)
    logits = model.forward_model(**args)  # [1, M, MAX_VOCAB]
    logits = logits[0]  # [M, MAX_VOCAB]
    out = {}
    offset = 0
    for m, positions in target_mods.items():
        n = len(positions)
        out[m] = logits[offset:offset + n]
        offset += n
    return out


def match_to_class_canonical(text: str) -> str | None:
    """Map free-form predicted text to one of CLASSES_CANONICAL or return None."""
    t = (text or "").lower().strip()
    if not t:
        return None
    # Direct: animal name as first word
    first = t.split()[0]
    for canon, short in zip(CLASSES_CANONICAL, CLASSES_SHORT):
        if short in t:
            return canon
        if first.startswith(short) or short.startswith(first):
            return canon
    return None


def cell(name):
    """Decorator: run a function, time it, swallow exceptions, save partial."""
    def wrap(fn):
        def runner(*a, **kw):
            print(f"\n{'='*60}\n  CELL: {name}\n{'='*60}")
            import time
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                print(f"  [OK] cell '{name}' in {time.time()-t0:.1f}s")
                return out
            except Exception as e:
                print(f"  [FAIL] cell '{name}': {type(e).__name__}: {e}")
                traceback.print_exc()
                return None
        return runner
    return wrap


# %% cell-2 memorization audio-suffix prediction -------------------------------
@cell("memorization_check_audio")
def cell2(n_samples=30, mask_frac=0.2):
    """80%-prefix audio + RGB + caption → predict 20%-suffix audio.
    Compare token-level accuracy on train vs test."""
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]  # 512
    RGB_LEN   = MAX_SEQ_LENS[MOD_TO_IDX["tok_rgb@196"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]

    def run(ds, indices):
        out = []
        for idx in indices:
            sample = get_modality_tokens(ds, idx)
            audio = sample["tok_audio@512"]
            cut = int((1 - mask_frac) * AUDIO_LEN)
            cut -= cut % 2  # codebook-aligned
            target = audio[cut:].to(DEVICE)
            logits = forward_logits(
                input_mods={
                    "tok_audio@512": (audio[:cut], list(range(cut))),
                    "tok_rgb@196":   (sample["tok_rgb@196"], list(range(RGB_LEN))),
                    "scene_desc":    (sample["scene_desc"],  list(range(SD_LEN))),
                },
                target_mods={"tok_audio@512": list(range(cut, AUDIO_LEN))},
            )["tok_audio@512"]
            # Restrict to audio vocab [0, 2048)
            audio_logits = logits[..., :VOCAB_SIZES[MOD_TO_IDX["tok_audio@512"]]]
            pred = audio_logits.argmax(dim=-1)
            acc = (pred == target).float().mean().item()
            ce = F.cross_entropy(audio_logits.float(), target).item()
            out.append({"stem": sample["__stem__"], "class": sample["__class__"],
                        "acc": acc, "ce": ce})
        return out

    train_idx = list(range(min(n_samples, len(train_ds))))
    test_idx  = list(range(min(n_samples, len(test_ds))))
    train_res = run(train_ds, train_idx)
    test_res  = run(test_ds,  test_idx)

    summary = {
        "train_acc": float(np.mean([r["acc"] for r in train_res])),
        "test_acc":  float(np.mean([r["acc"] for r in test_res])),
        "train_ce":  float(np.mean([r["ce"]  for r in train_res])),
        "test_ce":   float(np.mean([r["ce"]  for r in test_res])),
    }
    summary["gap_acc"] = summary["train_acc"] - summary["test_acc"]
    summary["gap_ce"]  = summary["test_ce"]  - summary["train_ce"]

    print(f"Train: acc={summary['train_acc']:.1%}  CE={summary['train_ce']:.3f}")
    print(f"Test : acc={summary['test_acc']:.1%}  CE={summary['test_ce']:.3f}")
    print(f"Gap  : {summary['gap_acc']:.1%} acc, {summary['gap_ce']:+.3f} CE")

    (RESULTS_DIR / "memorization_check.json").write_text(json.dumps({
        "train": train_res, "test": test_res, "summary": summary}, indent=2))
    return summary


# %% cell-3 audio → scene_desc classification ----------------------------------
@cell("audio_to_caption")
def cell3(n_samples=None):
    """Single forward pass: audio-only → predict scene_desc tokens → decode → class."""
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]
    n_samples = n_samples or len(test_ds)

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    correct, per_sample = 0, []
    for idx in tqdm(range(n_samples), desc="audio→cap"):
        sample = get_modality_tokens(test_ds, idx)
        audio = sample["tok_audio@512"]
        true_canon = sample["__class__"]
        true_idx = CLASSES_CANONICAL.index(true_canon) if true_canon in CLASSES_CANONICAL else -1
        logits = forward_logits(
            input_mods={"tok_audio@512": (audio, list(range(AUDIO_LEN)))},
            target_mods={"scene_desc":    list(range(SD_LEN))},
        )["scene_desc"]
        # Slice to GPT-2 BPE range only; tokens 50257-50303 are PAD/SOS/EOS or
        # train-time padding garbage that the tokenizer can't decode.
        pred_tokens = logits[..., :50257].argmax(dim=-1).cpu().tolist()
        pred_text = gpt2_tok.decode(pred_tokens, skip_special_tokens=True).strip()
        pred_canon = match_to_class_canonical(pred_text)
        pred_idx = CLASSES_CANONICAL.index(pred_canon) if pred_canon else -1
        if pred_idx == true_idx and true_idx >= 0:
            correct += 1
        if pred_idx >= 0 and true_idx >= 0:
            confusion[true_idx, pred_idx] += 1
        per_sample.append({"stem": sample["__stem__"], "true": true_canon,
                           "pred": pred_canon, "pred_text": pred_text,
                           "correct": pred_idx == true_idx})

    accuracy = correct / n_samples
    per_cls_acc = {}
    for i, c in enumerate(CLASSES_CANONICAL):
        row = confusion[i]
        per_cls_acc[c] = float(row[i] / max(row.sum(), 1))

    print(f"Top-1 accuracy: {accuracy:.1%}  (random {1/N_CLASSES:.1%})")
    print(f"Lift over random: {accuracy / (1/N_CLASSES):.1f}x")
    for c, a in sorted(per_cls_acc.items(), key=lambda x: -x[1]):
        print(f"  {c:24s} {a:.1%}")

    np.save(RESULTS_DIR / "audio2caption_confusion.npy", confusion)
    (RESULTS_DIR / "audio2caption.json").write_text(json.dumps({
        "accuracy": accuracy,
        "random_baseline": 1/N_CLASSES,
        "per_class_accuracy": per_cls_acc,
        "per_sample": per_sample[:200],   # cap for size
    }, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES_SHORT, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES_SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Audio → Caption (top-1={accuracy:.1%}, random={1/N_CLASSES:.1%})")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if confusion[i, j] > 0:
                color = "white" if confusion[i, j] > confusion.max() / 2 else "black"
                ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                        color=color, fontsize=8)
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(FIG_DIR / "audio2caption_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    return {"accuracy": accuracy, "per_class": per_cls_acc}


# %% cell-4 cross-modal retrieval R@K -----------------------------------------
@cell("cross_modal_retrieval")
def cell4(n_candidates=200):
    """For (audio_query, rgb_db): for each query find rank of paired image.
    We use the model's encoder hidden mean as the embedding."""
    @torch.no_grad()
    def encode_one(tokens: torch.Tensor, modality: str) -> torch.Tensor:
        L = tokens.shape[0]
        positions = torch.arange(L, device=DEVICE)
        mods = torch.full((1, L), MOD_TO_IDX[modality], dtype=torch.long, device=DEVICE)
        toks = tokens.unsqueeze(0).to(DEVICE)
        pos = positions.unsqueeze(0)
        pad = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
        x, _ = model.forward_encoder(toks, mods, pos, pad)  # [1, L, D]
        return x.mean(dim=1).squeeze(0)  # [D]

    indices = np.random.RandomState(42).choice(
        len(test_ds), min(n_candidates, len(test_ds)), replace=False)

    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    RGB_LEN   = MAX_SEQ_LENS[MOD_TO_IDX["tok_rgb@196"]]
    DEPTH_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_depth@196"]]
    LENMAP = {"tok_audio@512": AUDIO_LEN, "tok_rgb@196": RGB_LEN, "tok_depth@196": DEPTH_LEN}

    cache = {}
    for i in tqdm(indices, desc="encoding"):
        s = get_modality_tokens(test_ds, int(i))
        for m in ("tok_audio@512", "tok_rgb@196", "tok_depth@196"):
            cache.setdefault(m, []).append(encode_one(s[m], m))
    embs = {m: torch.stack(v) for m, v in cache.items()}
    embs = {m: v / v.norm(dim=-1, keepdim=True) for m, v in embs.items()}

    out = {}
    for src, dst in [("tok_audio@512", "tok_rgb@196"),
                     ("tok_rgb@196",   "tok_audio@512"),
                     ("tok_audio@512", "tok_depth@196"),
                     ("tok_depth@196", "tok_audio@512")]:
        sims = embs[src] @ embs[dst].t()
        N = sims.shape[0]
        ranks = []
        for i in range(N):
            sorted_idx = sims[i].argsort(descending=True)
            r = (sorted_idx == i).nonzero(as_tuple=True)[0].item()
            ranks.append(r)
        R1  = sum(r == 0 for r in ranks) / N
        R5  = sum(r <  5 for r in ranks) / N
        R10 = sum(r <  10 for r in ranks) / N
        key = f"{src.replace('tok_','').replace('@','at')}->{dst.replace('tok_','').replace('@','at')}"
        out[key] = {"R1": R1, "R5": R5, "R10": R10, "n_candidates": N}
        print(f"{src} → {dst}: R@1={R1:.1%} R@5={R5:.1%} R@10={R10:.1%}")

    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(out, indent=2))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(out.keys())
    x = np.arange(len(keys)); w = 0.25
    ax.bar(x - w, [out[k]["R1"]  for k in keys], w, label="R@1")
    ax.bar(x,     [out[k]["R5"]  for k in keys], w, label="R@5")
    ax.bar(x + w, [out[k]["R10"] for k in keys], w, label="R@10")
    ax.axhline(1/n_candidates, color="r", linestyle="--",
               label=f"R@1 random (1/{n_candidates})")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title(f"Cross-modal retrieval (n={n_candidates})")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "retrieval_barchart.png", dpi=150)
    plt.show()
    return out


# %% cell-5 audio-conditioned generation grid (MaskGIT) ------------------------
def maskgit_generate(input_mods, target_mod, n_iter=8, temperature=1.0):
    """Iterative MaskGIT sampling for `target_mod` conditioned on `input_mods`.
    Tokens are progressively unmasked from highest-confidence positions."""
    n_tokens = MAX_SEQ_LENS[MOD_TO_IDX[target_mod]]
    target_vocab = VOCAB_SIZES[MOD_TO_IDX[target_mod]]
    cur_tokens = torch.zeros(n_tokens, dtype=torch.long, device=DEVICE)
    cur_mask   = torch.zeros(n_tokens, dtype=torch.bool, device=DEVICE)  # True = known
    for it in range(n_iter):
        # Predict all positions
        local_inputs = dict(input_mods)
        if cur_mask.any():
            known_pos = cur_mask.nonzero(as_tuple=True)[0].tolist()
            local_inputs[target_mod] = (cur_tokens[cur_mask], known_pos)
        unknown_pos = (~cur_mask).nonzero(as_tuple=True)[0].tolist()
        if not unknown_pos:
            break
        logits = forward_logits(local_inputs, {target_mod: unknown_pos})[target_mod]
        logits = logits[..., :target_vocab].float()
        T = max(temperature * (1.0 - it / max(n_iter - 1, 1)), 1e-3)
        probs = torch.softmax(logits / T, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)  # [len(unknown_pos)]
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        # Keep ratio of known positions = cos schedule
        ratio_unknown = np.cos(np.pi / 2 * (it + 1) / n_iter)
        n_keep = max(1, int((1.0 - ratio_unknown) * n_tokens) - cur_mask.sum().item())
        n_keep = min(n_keep, conf.shape[0])
        top = conf.topk(n_keep).indices
        for j in top.tolist():
            p = unknown_pos[j]
            cur_tokens[p] = sampled[j]
            cur_mask[p] = True
    # Last pass — fill all
    if not cur_mask.all():
        unknown_pos = (~cur_mask).nonzero(as_tuple=True)[0].tolist()
        local_inputs = dict(input_mods)
        if cur_mask.any():
            known_pos = cur_mask.nonzero(as_tuple=True)[0].tolist()
            local_inputs[target_mod] = (cur_tokens[cur_mask], known_pos)
        logits = forward_logits(local_inputs, {target_mod: unknown_pos})[target_mod]
        logits = logits[..., :target_vocab]
        for j, p in enumerate(unknown_pos):
            cur_tokens[p] = logits[j].argmax()
            cur_mask[p] = True
    return cur_tokens


@cell("audio_conditioned_generation")
def cell5(n_show=10, maskgit_iter=8):
    if not DETOK_OK:
        print("skipping: detokenizers unavailable")
        return None
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]

    sel = np.linspace(0, len(test_ds)-1, n_show, dtype=int)
    samples = []
    for idx in tqdm(sel, desc="generate"):
        s = get_modality_tokens(test_ds, int(idx))
        audio = s["tok_audio@512"]
        # caption single-step
        cap_logits = forward_logits(
            input_mods={"tok_audio@512": (audio, list(range(AUDIO_LEN)))},
            target_mods={"scene_desc": list(range(SD_LEN))},
        )["scene_desc"]
        cap_tokens_bpe = cap_logits[..., :50257].argmax(dim=-1)
        cap_tokens_full = cap_logits.argmax(dim=-1)  # for re-conditioning, keep full
        cap_text = gpt2_tok.decode(
            cap_tokens_bpe.cpu().tolist(), skip_special_tokens=True).strip()
        # RGB MaskGIT
        rgb_tokens = maskgit_generate(
            {"tok_audio@512": (audio, list(range(AUDIO_LEN))),
             "scene_desc":    (cap_tokens_full, list(range(SD_LEN)))},
            target_mod="tok_rgb@196", n_iter=maskgit_iter,
        )
        # Depth MaskGIT (conditioned on audio+caption+rgb)
        depth_tokens = maskgit_generate(
            {"tok_audio@512": (audio, list(range(AUDIO_LEN))),
             "scene_desc":    (cap_tokens_full, list(range(SD_LEN))),
             "tok_rgb@196":   (rgb_tokens, list(range(MAX_SEQ_LENS[MOD_TO_IDX["tok_rgb@196"]])))},
            target_mod="tok_depth@196", n_iter=maskgit_iter,
        )
        # Decode tokens → images (DiVAE expects [B, H, W] of ids; 196 → 14x14)
        with torch.no_grad():
            rgb_img = rgb_detok.decode_tokens(rgb_tokens.view(1, 14, 14))[0].clamp(-1, 1)
            depth_img = depth_detok.decode_tokens(depth_tokens.view(1, 14, 14))[0]
            gt_rgb = rgb_detok.decode_tokens(s["tok_rgb@196"].to(DEVICE).view(1, 14, 14))[0].clamp(-1, 1)
        samples.append({
            "stem": s["__stem__"], "true_class": s["__class__"],
            "pred_caption": cap_text,
            "rgb_pred": rgb_img.cpu(), "depth_pred": depth_img.cpu(),
            "gt_rgb": gt_rgb.cpu(),
        })

    def to_img01(t):
        # DiVAE outputs in [-1, 1] for RGB
        return (t.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    fig, axes = plt.subplots(4, n_show, figsize=(2.2 * n_show, 9))
    for col, g in enumerate(samples):
        axes[0, col].imshow(to_img01(g["gt_rgb"]))
        axes[0, col].set_title(f"GT: {g['true_class'].split()[0]}", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].text(0.5, 0.5, f"Audio→\n'{g['pred_caption']}'",
                          ha="center", va="center", fontsize=9,
                          transform=axes[1, col].transAxes)
        axes[1, col].axis("off")
        axes[2, col].imshow(to_img01(g["rgb_pred"]))
        axes[2, col].axis("off")
        d = g["depth_pred"].squeeze().numpy()
        axes[3, col].imshow(d, cmap="viridis"); axes[3, col].axis("off")
    for row, lbl in enumerate(("GT RGB", "Pred caption", "Audio→RGB", "Audio→Depth")):
        axes[row, 0].text(-0.3, 0.5, lbl, rotation=90, va="center",
                          transform=axes[row, 0].transAxes, fontweight="bold")
    plt.suptitle("Audio-conditioned cross-modal generation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "audio_conditioned_generation_grid.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Individual samples
    sample_dir = FIG_DIR / "audio_conditioned_samples"; sample_dir.mkdir(exist_ok=True)
    for i, g in enumerate(samples):
        f, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(to_img01(g["gt_rgb"])); ax[0].set_title("GT"); ax[0].axis("off")
        ax[1].imshow(to_img01(g["rgb_pred"])); ax[1].set_title(f"Pred: {g['pred_caption']}"); ax[1].axis("off")
        ax[2].imshow(g["depth_pred"].squeeze().numpy(), cmap="viridis"); ax[2].set_title("Depth"); ax[2].axis("off")
        plt.savefig(sample_dir / f"sample_{i:02d}_{g['true_class'].split()[0]}.png", dpi=120, bbox_inches="tight")
        plt.show()
    return [{"stem": g["stem"], "true": g["true_class"], "pred": g["pred_caption"]}
            for g in samples], samples


# %% cell-6 external classifier (ResNet50/ImageNet) ----------------------------
@cell("external_classifier")
def cell6(gen_samples, n_full=80):
    if gen_samples is None:
        print("skipping: cell5 produced nothing")
        return None
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]

    import torchvision.models as tvm
    from torchvision import transforms
    classifier = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    IMAGENET_MAP = {
        "cat meowing":      list(range(281, 286)),
        "dog barking":      list(range(151, 269)),
        "chicken clucking": [7, 8],
        "cow lowing":       [345, 346],
        "coyote howling":   [272],
        "duck quacking":    [97],
        "horse neighing":   [339, 340],
        "lions roaring":    [291],
        "pig oinking":      [341],
        "sheep bleating":   [348, 349, 350],
        "pigeon cooing":    [16, 17],  # any small bird as proxy
    }

    def classify(rgb_chw01: torch.Tensor, true_canon: str):
        img = rgb_chw01.clamp(0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(normalize(img))
        top5 = logits[0].topk(5).indices.cpu().tolist()
        expected = IMAGENET_MAP.get(true_canon, [])
        return any(i in expected for i in top5), top5

    # Reuse the 10 from cell 5
    print(f"\n-- gen_samples (n={len(gen_samples[1])}) external check --")
    for g in gen_samples[1]:
        # DiVAE outputs [-1, 1] → convert to [0, 1]
        img01 = g["rgb_pred"] * 0.5 + 0.5
        hit, top5 = classify(img01, g["true_class"])
        g["external_top5_hit"] = bool(hit)
        print(f"  {g['stem']:30s} true={g['true_class']:20s} hit_top5={hit}")

    # Scale: generate + classify N more random test samples
    print(f"\n-- generating + classifying {n_full} extra test samples --")
    extra_indices = np.random.RandomState(7).choice(len(test_ds), n_full, replace=False)
    results = []
    for idx in tqdm(extra_indices, desc="extern-eval"):
        s = get_modality_tokens(test_ds, int(idx))
        audio = s["tok_audio@512"]
        cap_logits = forward_logits(
            {"tok_audio@512": (audio, list(range(AUDIO_LEN)))},
            {"scene_desc": list(range(SD_LEN))},
        )["scene_desc"]
        cap_tokens = cap_logits.argmax(dim=-1)
        rgb_tokens = maskgit_generate(
            {"tok_audio@512": (audio, list(range(AUDIO_LEN))),
             "scene_desc":    (cap_tokens, list(range(SD_LEN)))},
            target_mod="tok_rgb@196", n_iter=4,  # faster: 4 iter
        )
        with torch.no_grad():
            rgb = rgb_detok.decode_tokens(rgb_tokens.view(1, 14, 14))[0].cpu()
        img01 = rgb * 0.5 + 0.5
        hit, _ = classify(img01, s["__class__"])
        results.append({"stem": s["__stem__"], "true_class": s["__class__"], "hit": bool(hit)})

    hit_rate = sum(r["hit"] for r in results) / max(1, len(results))
    print(f"\nExternal classifier top-5 hit rate (n={n_full}): {hit_rate:.1%}")
    per_cls = defaultdict(lambda: {"hit": 0, "total": 0})
    for r in results:
        per_cls[r["true_class"]]["total"] += 1
        per_cls[r["true_class"]]["hit"]   += int(r["hit"])
    for c in CLASSES_CANONICAL:
        p = per_cls[c]
        rate = p["hit"] / p["total"] if p["total"] else 0
        print(f"  {c:24s} {p['hit']:3d}/{p['total']:3d} = {rate:.1%}")

    (RESULTS_DIR / "external_validation.json").write_text(json.dumps({
        "hit_rate_top5": hit_rate,
        "per_class": {c: dict(per_cls[c]) for c in CLASSES_CANONICAL},
        "per_sample": results,
    }, indent=2))
    return {"hit_rate": hit_rate}


# %% cell-7 forward vs autoregressive comparison -------------------------------
@cell("forward_vs_autoregressive")
def cell7(n_samples=50):
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]
    sd_vocab  = VOCAB_SIZES[MOD_TO_IDX["scene_desc"]]

    ces_single, ces_ar = [], []
    for idx in tqdm(range(min(n_samples, len(test_ds))), desc="single vs AR"):
        s = get_modality_tokens(test_ds, idx)
        audio = s["tok_audio@512"]
        target = s["scene_desc"].to(DEVICE)
        # Mode 1: single forward
        logits_single = forward_logits(
            {"tok_audio@512": (audio, list(range(AUDIO_LEN)))},
            {"scene_desc":    list(range(SD_LEN))},
        )["scene_desc"][..., :sd_vocab]
        ce_single = F.cross_entropy(logits_single.float(), target).item()
        ces_single.append(ce_single)
        # Mode 2: autoregressive
        ce_pos = []
        predicted = []
        for pos in range(SD_LEN):
            ins = {"tok_audio@512": (audio, list(range(AUDIO_LEN)))}
            if predicted:
                ins["scene_desc"] = (torch.tensor(predicted, dtype=torch.long),
                                     list(range(len(predicted))))
            logits = forward_logits(ins, {"scene_desc": [pos]})["scene_desc"][0, :sd_vocab]
            ce_pos.append(F.cross_entropy(logits.unsqueeze(0).float(),
                                          target[pos:pos+1]).item())
            predicted.append(int(logits.argmax()))
        ces_ar.append(float(np.mean(ce_pos)))

    mean_s = float(np.mean(ces_single)); mean_a = float(np.mean(ces_ar))
    gap = mean_a - mean_s
    print(f"single CE  : {mean_s:.3f}")
    print(f"AR CE      : {mean_a:.3f}")
    print(f"gap (a-s)  : {gap:+.3f} nats")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Single forward\n(training-aligned)", "Autoregressive"],
                  [mean_s, mean_a], color=["#2ecc71", "#e74c3c"])
    for b, v in zip(bars, [mean_s, mean_a]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("CE (nats)")
    ax.set_title(f"Caption prediction: gap = {gap:+.3f} nats")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "forward_vs_ar.png", dpi=150)
    plt.show()
    (RESULTS_DIR / "forward_vs_ar.json").write_text(json.dumps({
        "n_samples": n_samples,
        "mean_single_forward_ce": mean_s,
        "mean_autoregressive_ce": mean_a,
        "gap": gap,
    }, indent=2))
    return {"single": mean_s, "ar": mean_a, "gap": gap}


# %% cell-8 summary placeholder (keeps return wiring for later) ----------------
def decode_audio_tokens(tokens: torch.Tensor) -> np.ndarray:
    """[512] tokens with cb2 offset → wav numpy array via EnCodec."""
    cb1 = tokens[0::2].clone()
    cb2 = tokens[1::2].clone() - 1024
    cb1 = cb1.clamp(0, 1023); cb2 = cb2.clamp(0, 1023)
    codes = torch.stack([cb1, cb2]).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, 2, 256]
    with torch.no_grad():
        wav = encodec.decode([(codes.squeeze(0), None)])[0].squeeze().cpu().numpy()
    return wav


@cell("audio_samples_and_summary")
def cell8(memo_summary, audio2cap_res, retrieval_res, extern_res, fa_res):
    """Audio-related results live in cell 9. This cell only writes the
    final summary JSON so the rest of the pipeline can find it."""
    if not ENC_OK:
        print("EnCodec unavailable — audio cells will be skipped")

    summary = {
        "dataset": {
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "n_classes": N_CLASSES,
            "classes_canonical": CLASSES_CANONICAL,
        },
        "training": {
            "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "checkpoint": CKPT_PATH,
        },
        "results": {},
    }
    if memo_summary:    summary["results"]["memorization"] = memo_summary
    if audio2cap_res:   summary["results"]["audio2caption"] = audio2cap_res
    if retrieval_res:   summary["results"]["retrieval"] = retrieval_res
    if extern_res:      summary["results"]["external_validation"] = extern_res
    if fa_res:          summary["results"]["forward_vs_ar"] = fa_res

    (RESULTS_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(json.dumps(summary["results"], indent=2))
    print("=" * 60)
    return summary


# %% cell-9 audio generation probes ------------------------------------------
# 9a: 80-20 audio suffix completion (Jason's "charitable" test)
# 9b: cross-modal generation from class label only (hardest)
# 9c: cross-modal generation from rgb+depth+normal+caption (intermediate)
# + metrics: RMS energy spread, ZCR, PANNs AudioSet classifier confidence
# + per-sample spectrograms inline

def compute_rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))


def compute_zcr(wav: np.ndarray) -> float:
    """Zero-crossing rate per sample (proxy for noise vs tonal content)."""
    if wav.size < 2:
        return 0.0
    sign_changes = np.sum(np.diff(np.sign(wav)) != 0)
    return float(sign_changes) / float(wav.size - 1)


def plot_spectrogram(ax, wav: np.ndarray, sr: int = 24000, title: str = ""):
    from scipy import signal
    if wav.size < 256:
        ax.set_title(f"{title} [too short]"); ax.axis("off"); return
    f, t, Sxx = signal.spectrogram(wav, sr, nperseg=512, noverlap=256)
    ax.pcolormesh(t, f, np.log10(Sxx + 1e-10), shading="auto", cmap="magma")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(50, sr / 2)
    ax.set_ylabel("Hz"); ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=10)


def load_pann_class_map():
    """Map our canonical classes → AudioSet PANN indices via keyword match
    on class_labels_indices.csv. Returns {class: [indices]} and the loaded
    PANN model. If PANNs is unavailable, returns (None, None)."""
    try:
        from panns_inference import AudioTagging
    except Exception as e:
        print(f"PANNs unavailable: {e}")
        return None, None
    import csv
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not csv_path.exists():
        print(f"missing {csv_path}")
        return None, None
    labels = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                labels[int(row[0])] = row[2].strip('"').lower()
    keywords = {
        "cat meowing":      ["cat", "meow", "purr", "caterwaul"],
        "chicken clucking": ["chicken", "cluck", "rooster", "crow"],
        "cow lowing":       ["cattle", "moo", "bovinae"],
        "coyote howling":   ["coyote", "howl", "wolf"],
        "dog barking":      ["dog", "bark", "yip", "growl"],
        "duck quacking":    ["duck", "quack"],
        "horse neighing":   ["horse", "neigh", "whinny", "clip-clop"],
        "lions roaring":    ["lion", "roar"],
        "pig oinking":      ["pig", "oink"],
        "sheep bleating":   ["sheep", "bleat"],
        "pigeon cooing":    ["pigeon", "dove", "coo"],
    }
    pann_map = {}
    for cls, kws in keywords.items():
        ids = [idx for idx, name in labels.items()
               if any(kw in name for kw in kws)]
        pann_map[cls] = sorted(set(ids))
    print("PANN AudioSet → class mapping:")
    for cls, ids in pann_map.items():
        names = [labels[i] for i in ids]
        print(f"  {cls:24s} indices={ids}  labels={names}")
    at = AudioTagging(checkpoint_path=None, device=DEVICE)
    return pann_map, at


def score_audio_pann(at, wav: np.ndarray, sr: int = 24000,
                    target_indices: list = None):
    """Return (max_score_over_target, top_idx, top_label_global_idx).
    Resamples 24kHz → 32kHz (PANN's expected rate) via scipy.signal.resample."""
    from scipy.signal import resample
    n_out = int(wav.size * 32000 / sr)
    wav32 = resample(wav.astype(np.float32), n_out).astype(np.float32)
    if wav32.size < 32000:
        wav32 = np.pad(wav32, (0, 32000 - wav32.size))
    arr = wav32[None, :].astype(np.float32)
    clipwise, _ = at.inference(arr)
    scores = clipwise[0]
    target_score = float(scores[target_indices].max()) if target_indices else 0.0
    top_idx = int(np.argmax(scores))
    return target_score, top_idx, float(scores[top_idx])


@cell("audio_generation_probes")
def cell9(n_samples=5, n_iter=8):
    """Three audio-generation probes + per-class metrics + spectrograms."""
    from IPython.display import display, Audio, Markdown
    import soundfile as sf
    AUDIO_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_audio@512"]]
    SD_LEN    = MAX_SEQ_LENS[MOD_TO_IDX["scene_desc"]]
    RGB_LEN   = MAX_SEQ_LENS[MOD_TO_IDX["tok_rgb@196"]]
    DEPTH_LEN = MAX_SEQ_LENS[MOD_TO_IDX["tok_depth@196"]]
    NORM_LEN  = MAX_SEQ_LENS[MOD_TO_IDX["tok_normal@196"]]
    audio_dir = FIG_DIR / "audio_demos"; audio_dir.mkdir(exist_ok=True)

    if not ENC_OK:
        print("EnCodec unavailable — skipping audio generation cell")
        return None

    pann_map, pann = load_pann_class_map()

    # Pick one sample per class for 9a + 9c — most informative diversity
    class_to_idx = defaultdict(list)
    for i in range(len(test_ds)):
        c = read_class_canonical("test", test_ds.file_names[i])
        if c in CLASSES_CANONICAL:
            class_to_idx[c].append(i)
    sample_indices = {c: ids[0] for c, ids in class_to_idx.items() if ids}

    results = {"9a": {}, "9b": {}, "9c": {}}
    spectrogram_grid = {"9a": [], "9b": [], "9c": []}

    # =========================================================
    # 9a) 80-20 audio suffix completion
    # =========================================================
    print("\n" + "=" * 60)
    print("CELL 9a — Audio suffix completion (80% prefix + others → 20% suffix)")
    print("=" * 60)
    display(Markdown("## 9a) Audio suffix completion — *Jason's '80-20' test*"))
    display(Markdown("**Input**: 80% audio prefix + GT RGB + depth + normal + caption  \n"
                     "**Output**: model predicts 20% audio suffix → decode → wav"))
    cut = int(0.8 * AUDIO_LEN); cut -= cut % 2  # codebook-aligned

    for cls, idx in list(sample_indices.items()):
        s = get_modality_tokens(test_ds, idx)
        # Generate suffix only — input = audio prefix + all visual + caption
        gen_suffix = maskgit_generate(
            input_mods={
                "tok_audio@512":  (s["tok_audio@512"][:cut], list(range(cut))),
                "tok_rgb@196":    (s["tok_rgb@196"],    list(range(RGB_LEN))),
                "tok_depth@196":  (s["tok_depth@196"],  list(range(DEPTH_LEN))),
                "tok_normal@196": (s["tok_normal@196"], list(range(NORM_LEN))),
                "scene_desc":     (s["scene_desc"],    list(range(SD_LEN))),
            },
            target_mod="tok_audio@512", n_iter=n_iter, temperature=1.0,
        )
        # The MaskGIT call returns ALL 512 positions filled; take only [cut, 512)
        # as the "generated" suffix, keep prefix as GT.
        full = s["tok_audio@512"].to(DEVICE).clone()
        full[cut:] = gen_suffix[cut:]
        wav_gen = decode_audio_tokens(full)
        wav_gt  = decode_audio_tokens(s["tok_audio@512"])
        rms_gen, rms_gt = compute_rms(wav_gen), compute_rms(wav_gt)
        zcr_gen, zcr_gt = compute_zcr(wav_gen), compute_zcr(wav_gt)
        pann_score = None
        if pann is not None and cls in pann_map and pann_map[cls]:
            pann_score, _, _ = score_audio_pann(pann, wav_gen, target_indices=pann_map[cls])
        short = cls.split()[0]
        sf.write(audio_dir / f"9a_{short}_gen.wav", wav_gen, 24000)
        sf.write(audio_dir / f"9a_{short}_gt.wav",  wav_gt,  24000)
        results["9a"][cls] = {
            "rms_gen": rms_gen, "rms_gt": rms_gt,
            "zcr_gen": zcr_gen, "zcr_gt": zcr_gt,
            "pann_score": pann_score,
        }
        display(Markdown(f"#### {cls}  (stem `{s['__stem__']}`)"))
        display(Markdown(
            f"GT — rms={rms_gt:.4f}, zcr={zcr_gt:.4f}  &nbsp;&nbsp; "
            f"Gen — rms={rms_gen:.4f}, zcr={zcr_gen:.4f}"
            + (f"  &nbsp;&nbsp; PANN target-score: **{pann_score:.3f}**" if pann_score is not None else "")
        ))
        display(Markdown("**Ground truth audio**:"))
        display(Audio(wav_gt, rate=24000))
        display(Markdown("**Generated (80-20 completion)**:"))
        display(Audio(wav_gen, rate=24000))
        spectrogram_grid["9a"].append((cls, wav_gt, wav_gen))

    # Spectrogram grid for 9a
    fig, axes = plt.subplots(len(spectrogram_grid["9a"]), 2, figsize=(11, 2.0 * len(spectrogram_grid["9a"])))
    if len(spectrogram_grid["9a"]) == 1:
        axes = axes[None, :]
    for r, (cls, wav_gt, wav_gen) in enumerate(spectrogram_grid["9a"]):
        plot_spectrogram(axes[r, 0], wav_gt,  title=f"{cls}  — GT")
        plot_spectrogram(axes[r, 1], wav_gen, title=f"{cls}  — 9a generated suffix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "9a_spectrograms.png", dpi=140, bbox_inches="tight")
    plt.show()

    # =========================================================
    # 9b) Cross-modal generation: class label only → audio
    # =========================================================
    print("\n" + "=" * 60)
    print("CELL 9b — Audio generation from class label only (hardest test)")
    print("=" * 60)
    display(Markdown("## 9b) Pure cross-modal: class label → audio"))
    display(Markdown("**Input**: scene_desc only (e.g. `dog barking`)  \n"
                     "**Output**: 512 audio tokens via MaskGIT → decode → wav"))

    for cls in CLASSES_CANONICAL:
        tokenized = test_ds.text_tokenizer(
            cls, max_length=test_ds.text_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )["input_ids"][0].long()
        gen_tokens = maskgit_generate(
            input_mods={"scene_desc": (tokenized, list(range(SD_LEN)))},
            target_mod="tok_audio@512", n_iter=n_iter, temperature=1.0,
        )
        wav_gen = decode_audio_tokens(gen_tokens)
        rms_gen = compute_rms(wav_gen); zcr_gen = compute_zcr(wav_gen)
        pann_score = None; pann_top_label = None
        if pann is not None and cls in pann_map and pann_map[cls]:
            pann_score, _, _ = score_audio_pann(pann, wav_gen, target_indices=pann_map[cls])
        short = cls.split()[0]
        sf.write(audio_dir / f"9b_class_{short}.wav", wav_gen, 24000)
        results["9b"][cls] = {"rms": rms_gen, "zcr": zcr_gen, "pann_score": pann_score}
        display(Markdown(f"#### {cls}"))
        display(Markdown(
            f"rms={rms_gen:.4f}  zcr={zcr_gen:.4f}"
            + (f"  &nbsp;&nbsp; PANN target-score: **{pann_score:.3f}**"
               if pann_score is not None else "")
        ))
        display(Audio(wav_gen, rate=24000))
        spectrogram_grid["9b"].append((cls, wav_gen))

    fig, axes = plt.subplots(len(spectrogram_grid["9b"]), 1, figsize=(8, 1.6 * len(spectrogram_grid["9b"])))
    if len(spectrogram_grid["9b"]) == 1:
        axes = [axes]
    for r, (cls, wav) in enumerate(spectrogram_grid["9b"]):
        plot_spectrogram(axes[r], wav, title=f"9b — {cls} (class-only generation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "9b_spectrograms.png", dpi=140, bbox_inches="tight")
    plt.show()

    # =========================================================
    # 9c) Multi-modal generation: rgb+depth+normal+caption → audio
    # =========================================================
    print("\n" + "=" * 60)
    print("CELL 9c — Audio generation from rgb+depth+normal+caption (intermediate)")
    print("=" * 60)
    display(Markdown("## 9c) Intermediate cross-modal: visual context → audio"))
    display(Markdown("**Input**: RGB + depth + normal + caption  \n"
                     "**Output**: 512 audio tokens via MaskGIT → decode → wav"))

    for cls, idx in list(sample_indices.items()):
        s = get_modality_tokens(test_ds, idx)
        gen_tokens = maskgit_generate(
            input_mods={
                "tok_rgb@196":    (s["tok_rgb@196"],    list(range(RGB_LEN))),
                "tok_depth@196":  (s["tok_depth@196"],  list(range(DEPTH_LEN))),
                "tok_normal@196": (s["tok_normal@196"], list(range(NORM_LEN))),
                "scene_desc":     (s["scene_desc"],     list(range(SD_LEN))),
            },
            target_mod="tok_audio@512", n_iter=n_iter, temperature=1.0,
        )
        wav_gen = decode_audio_tokens(gen_tokens)
        rms_gen = compute_rms(wav_gen); zcr_gen = compute_zcr(wav_gen)
        pann_score = None
        if pann is not None and cls in pann_map and pann_map[cls]:
            pann_score, _, _ = score_audio_pann(pann, wav_gen, target_indices=pann_map[cls])
        short = cls.split()[0]
        sf.write(audio_dir / f"9c_visual_{short}.wav", wav_gen, 24000)
        results["9c"][cls] = {"rms": rms_gen, "zcr": zcr_gen, "pann_score": pann_score}
        display(Markdown(f"#### {cls}  (stem `{s['__stem__']}`)"))
        display(Markdown(
            f"rms={rms_gen:.4f}  zcr={zcr_gen:.4f}"
            + (f"  &nbsp;&nbsp; PANN target-score: **{pann_score:.3f}**"
               if pann_score is not None else "")
        ))
        display(Audio(wav_gen, rate=24000))
        spectrogram_grid["9c"].append((cls, wav_gen))

    fig, axes = plt.subplots(len(spectrogram_grid["9c"]), 1, figsize=(8, 1.6 * len(spectrogram_grid["9c"])))
    if len(spectrogram_grid["9c"]) == 1:
        axes = [axes]
    for r, (cls, wav) in enumerate(spectrogram_grid["9c"]):
        plot_spectrogram(axes[r], wav, title=f"9c — {cls} (visual+text generation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "9c_spectrograms.png", dpi=140, bbox_inches="tight")
    plt.show()

    # =========================================================
    # Diversity / mode-collapse summary
    # =========================================================
    print("\n" + "=" * 60)
    print("AUDIO GEN SUMMARY (RMS spread, PANN target-class confidence)")
    print("=" * 60)
    rms_b = [results["9b"][c]["rms"] for c in CLASSES_CANONICAL]
    rms_c = [results["9c"][c]["rms"] for c in CLASSES_CANONICAL]
    rms_gt = [results["9a"][c]["rms_gt"] for c in CLASSES_CANONICAL if c in results["9a"]]
    spread_gt = float(np.std(rms_gt)) if rms_gt else 0.0
    spread_b  = float(np.std(rms_b))
    spread_c  = float(np.std(rms_c))
    print(f"RMS std (GT)      : {spread_gt:.5f}")
    print(f"RMS std (9b class): {spread_b:.5f}   ratio vs GT: {spread_b/max(spread_gt,1e-9):.3f}")
    print(f"RMS std (9c multi): {spread_c:.5f}   ratio vs GT: {spread_c/max(spread_gt,1e-9):.3f}")

    if pann is not None:
        pann_b_hits = sum(1 for c in CLASSES_CANONICAL
                          if results["9b"][c]["pann_score"] is not None
                          and results["9b"][c]["pann_score"] > 0.30)
        pann_c_hits = sum(1 for c in CLASSES_CANONICAL
                          if results["9c"][c]["pann_score"] is not None
                          and results["9c"][c]["pann_score"] > 0.30)
        print(f"PANN target-class > 0.30 hits  9b: {pann_b_hits}/{N_CLASSES}")
        print(f"PANN target-class > 0.30 hits  9c: {pann_c_hits}/{N_CLASSES}")

    results["summary"] = {
        "rms_std_gt": spread_gt, "rms_std_9b": spread_b, "rms_std_9c": spread_c,
        "rms_spread_ratio_9b_over_gt": spread_b / max(spread_gt, 1e-9),
        "rms_spread_ratio_9c_over_gt": spread_c / max(spread_gt, 1e-9),
    }
    (RESULTS_DIR / "audio_generation_probes.json").write_text(json.dumps(results, indent=2))
    return results


# %% run-cell-2 ---------------------------------------------------------------
memo_summary = cell2()

# %% run-cell-3 ---------------------------------------------------------------
audio2cap_res = cell3()

# %% run-cell-4 ---------------------------------------------------------------
retrieval_res = cell4()

# %% run-cell-5 ---------------------------------------------------------------
gen_samples = cell5()

# %% run-cell-6 ---------------------------------------------------------------
extern_res = cell6(gen_samples)

# %% run-cell-7 ---------------------------------------------------------------
fa_res = cell7()

# %% run-cell-8 ---------------------------------------------------------------
cell8(memo_summary, audio2cap_res, retrieval_res, extern_res, fa_res)

# %% run-cell-9 audio generation probes --------------------------------------
audio_gen_res = cell9()

# %% finalize -----------------------------------------------------------------
print(f"\nALL ARTIFACTS UNDER: {FIG_DIR.parent.resolve()}")
