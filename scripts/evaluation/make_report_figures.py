"""Report-ready figures for nano4M-Audio, generated from the latest checkpoint.

Figures
  1. Per-modality cross-entropy drop  (clean greyscale bar chart, LaTeX typography)
  2. Depth / Normal reconstruction grid  (RGB | GT depth | pred depth | GT normal | pred normal)
  3. Caption -> RGB gallery  (per-class, several samples)
  4. Audio -> RGB generation grid  (many candidates over temperature x seed, for selection)

Style: charts are pure greyscale with serif (Computer-Modern / LaTeX) typography and
minimal chrome. Sample images keep their intrinsic content (RGB photos in colour, depth
in grayscale, normal maps in their canonical RGB encoding -- greyscaling a normal map
would destroy its meaning). All outputs saved as vector PDF + 300-dpi PNG.

Run on a GPU node (see sbatch_make_report_figures.sh).
"""
from __future__ import annotations
import os, sys, json, math, argparse, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
NANO4M_DIR = "/path/to/nanofm"
DATA_ROOT  = "/path/to/nano4M-Audio/data/tokenized_v5"
CFG_PATH   = f"{NANO4M_DIR}/cfgs/nano4M/animal_full_5mod_v5.yaml"
CKPT_PATH  = "/path/to/nano4M-Audio/runs/animal_full_5mod_v5_fresh/checkpoint-final.safetensors"
OUT_DIR    = Path("/path/to/nano4M-Audio/eval/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "audio2rgb_candidates").mkdir(exist_ok=True)
(OUT_DIR / "caption2rgb_candidates").mkdir(exist_ok=True)

sys.path.insert(0, NANO4M_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0); torch.manual_seed(0)

CLASSES_CANONICAL = [
    "cat meowing", "chicken clucking", "cow lowing", "coyote howling",
    "dog barking", "duck quacking", "horse neighing", "lions roaring",
    "pig oinking", "sheep bleating", "pigeon cooing",
]
SHORT = {c: c.split()[0] for c in CLASSES_CANONICAL}

# ----------------------------------------------------------------------------
# Publication style: prefer real LaTeX, fall back to Computer-Modern mathtext.
# ----------------------------------------------------------------------------
def setup_style():
    base = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # editable text in PDF
        "ps.fonttype": 42,
    }
    use_tex = False
    try:
        import shutil
        if shutil.which("latex") and shutil.which("dvipng"):
            tex = {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
            plt.rcParams.update({**base, **tex})
            fig = plt.figure(); plt.text(0.5, 0.5, r"$\mathrm{test}\ 10.8$")
            fig.canvas.draw(); plt.close(fig)   # force a render to validate TeX
            use_tex = True
        else:
            raise RuntimeError("no system latex")
    except Exception as e:
        print(f"[style] system LaTeX unavailable ({e}); using Computer-Modern mathtext.")
        cm = {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
        }
        plt.rcParams.update({**base, **cm})
    print(f"[style] text.usetex = {use_tex}")
    return use_tex

# Greyscale palette
GREY = {"baseline": "#d9d9d9", "bar": "#3a3a3a", "edge": "#000000",
        "mid": "#888888", "light": "#bfbfbf"}

# ----------------------------------------------------------------------------
# Load cfg, model, detokenizers, dataset
# ----------------------------------------------------------------------------
print("loading config + model ...", flush=True)
cfg = OmegaConf.load(CFG_PATH)
MODALITIES   = list(cfg.global_vars.modalities)
VOCAB_SIZES  = list(cfg.global_vars.vocab_sizes)
MAX_SEQ_LENS = list(cfg.global_vars.max_seq_lens)
MAX_VOCAB    = max(VOCAB_SIZES)
MOD_TO_IDX   = {m: i for i, m in enumerate(MODALITIES)}
SPAN_MODS    = OmegaConf.to_container(cfg.get("span_modalities", {})) or {}

from nanofm.models.fourm import FourM
from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset
from nanofm.data.multimodal import create_multimodal_masked_dataloader

model = instantiate(cfg.model_config).to(DEVICE).eval()
from safetensors.torch import load_file
state = load_file(CKPT_PATH, device="cpu")
state = {k.replace("module.", ""): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

from fourm.vq.vqvae import DiVAE
rgb_detok    = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").to(DEVICE).eval()
depth_detok  = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(DEVICE).eval()
normal_detok = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_normal_8k_224-448").to(DEVICE).eval()

def read_class(split, stem):
    return json.loads((Path(DATA_ROOT) / split / "scene_desc" / f"{stem}.json").read_text())[0]

test_ds = SimpleMultimodalDataset(
    root_dir=DATA_ROOT, split="test", modalities=MODALITIES, transforms=None,
    sample_from_k_augmentations=10, text_tokenizer_path="gpt2", text_max_length=64,
)
print(f"  test set: {len(test_ds)} stems", flush=True)

# ----------------------------------------------------------------------------
# Helpers (verbatim from the validated eval pipeline)
# ----------------------------------------------------------------------------
def get_modality_tokens(ds, idx, aug=None):
    fn = ds.file_names[idx]
    aug = np.random.randint(0, ds.sample_from_k_augmentations) if aug is None else aug
    out = {"__stem__": fn, "__class__": read_class("test", fn)}
    for mod in MODALITIES:
        ext = ds.modality_extensions[mod]
        path = Path(ds.root_dir) / ds.split / mod / f"{fn}{ext}"
        if "tok" in mod:
            out[mod] = torch.from_numpy(np.load(path)[aug]).long()
        else:
            caps = json.loads(path.read_text())
            tok = ds.text_tokenizer(caps[aug], max_length=ds.text_max_length,
                                    padding="max_length", truncation=True, return_tensors="pt")
            out[mod] = tok["input_ids"][0].long()
    return out

def build_eval_batch(input_mods, target_mods):
    et, em, ep = [], [], []
    for m, (toks, pos) in input_mods.items():
        pos = torch.as_tensor(pos, dtype=torch.long, device=DEVICE)
        toks = torch.as_tensor(toks, dtype=torch.long, device=DEVICE)
        et.append(toks); ep.append(pos); em.append(torch.full_like(pos, MOD_TO_IDX[m]))
    dm, dp = [], []
    for m, pos in target_mods.items():
        pos = torch.as_tensor(pos, dtype=torch.long, device=DEVICE)
        dp.append(pos); dm.append(torch.full_like(pos, MOD_TO_IDX[m]))
    et = torch.cat(et).unsqueeze(0); em = torch.cat(em).unsqueeze(0); ep = torch.cat(ep).unsqueeze(0)
    dm = torch.cat(dm).unsqueeze(0) if dm else torch.zeros(1, 0, dtype=torch.long, device=DEVICE)
    dp = torch.cat(dp).unsqueeze(0) if dp else torch.zeros(1, 0, dtype=torch.long, device=DEVICE)
    return dict(enc_input_tokens=et, enc_input_modalities=em, enc_input_positions=ep,
                dec_input_modalities=dm, dec_input_positions=dp,
                enc_pad_mask=torch.ones_like(et, dtype=torch.bool),
                dec_pad_mask=torch.ones_like(dm, dtype=torch.bool))

@torch.no_grad()
def forward_logits(input_mods, target_mods):
    args = build_eval_batch(input_mods, target_mods)
    logits = model.forward_model(**args)[0]
    out, off = {}, 0
    for m, pos in target_mods.items():
        out[m] = logits[off:off + len(pos)]; off += len(pos)
    return out

def _nucleus_sample(probs, top_p):
    """Top-p (nucleus) sampling over the last dim. probs: [N, V] (already softmaxed)."""
    if top_p <= 0 or top_p >= 1:
        return torch.multinomial(probs, 1).squeeze(-1)
    sp, si = torch.sort(probs, dim=-1, descending=True)
    csum = sp.cumsum(-1)
    keep = csum - sp <= top_p          # always keep the top token
    sp = sp * keep
    sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
    pick = torch.multinomial(sp, 1)
    return si.gather(-1, pick).squeeze(-1)

def maskgit_generate(input_mods, target_mod, n_iter=12, temperature=1.0,
                     temp_floor=None, top_p=0.0):
    """Iterative MaskGIT sampling. `temp_floor` keeps sampling stochastic at the
    end (avoids collapse-to-argmax so candidates differ); `top_p` adds nucleus
    sampling for diversity."""
    n_tokens = MAX_SEQ_LENS[MOD_TO_IDX[target_mod]]
    tv = VOCAB_SIZES[MOD_TO_IDX[target_mod]]
    floor = 1e-3 if temp_floor is None else temp_floor
    cur = torch.zeros(n_tokens, dtype=torch.long, device=DEVICE)
    known = torch.zeros(n_tokens, dtype=torch.bool, device=DEVICE)
    for it in range(n_iter):
        li = dict(input_mods)
        if known.any():
            kp = known.nonzero(as_tuple=True)[0].tolist()
            li[target_mod] = (cur[known], kp)
        up = (~known).nonzero(as_tuple=True)[0].tolist()
        if not up: break
        logits = forward_logits(li, {target_mod: up})[target_mod][..., :tv].float()
        T = max(temperature * (1.0 - it / max(n_iter - 1, 1)), floor)
        probs = torch.softmax(logits / T, dim=-1)
        samp = _nucleus_sample(probs, top_p)
        conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)
        ru = np.cos(np.pi / 2 * (it + 1) / n_iter)
        nkeep = max(1, int((1.0 - ru) * n_tokens) - known.sum().item())
        nkeep = min(nkeep, conf.shape[0])
        for j in conf.topk(nkeep).indices.tolist():
            cur[up[j]] = samp[j]; known[up[j]] = True
    if not known.all():
        up = (~known).nonzero(as_tuple=True)[0].tolist()
        li = dict(input_mods)
        if known.any():
            kp = known.nonzero(as_tuple=True)[0].tolist(); li[target_mod] = (cur[known], kp)
        logits = forward_logits(li, {target_mod: up})[target_mod][..., :tv].float()
        probs = torch.softmax(logits / max(floor, 1e-3), dim=-1)
        samp = _nucleus_sample(probs, top_p)
        for j, p in enumerate(up):
            cur[p] = samp[j]; known[p] = True
    return cur

@torch.no_grad()
def decode_rgb(tokens):
    img = rgb_detok.decode_tokens(tokens.view(1, 14, 14))[0].clamp(-1, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

@torch.no_grad()
def decode_depth(tokens):
    d = depth_detok.decode_tokens(tokens.view(1, 14, 14))[0].cpu()  # [C,H,W] or [H,W]
    if d.dim() == 3:
        d = d.mean(0)            # collapse channels -> [H,W]
    d = d.numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d  # grayscale [H,W]

@torch.no_grad()
def decode_normal(tokens):
    n = normal_detok.decode_tokens(tokens.view(1, 14, 14))[0].clamp(-1, 1)
    return (n.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)  # RGB-encoded

FULL = {m: list(range(MAX_SEQ_LENS[MOD_TO_IDX[m]])) for m in MODALITIES}


# ============================================================================
# FIGURE 1 — per-modality cross-entropy drop
# ============================================================================
def fig1_ce_drop(n_batches=12, batch_size=64):
    print("\n[fig1] computing per-modality CE on test set ...", flush=True)
    loader = create_multimodal_masked_dataloader(
        root_dir=DATA_ROOT, split="test", modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES, max_seq_lens=MAX_SEQ_LENS,
        input_alphas=list(cfg.global_vars.input_alphas),
        target_alphas=list(cfg.global_vars.target_alphas),
        input_tokens_range=list(cfg.global_vars.input_tokens_range),
        target_tokens_range=list(cfg.global_vars.target_tokens_range),
        overlap_vocab=True, overlap_posembs=True, sample_from_k_augmentations=10,
        text_tokenizer_path="gpt2", text_max_length=64, batch_size=batch_size,
        infinite=False, num_workers=4, pin_memory=True, shuffle=True,
        drop_last=True, distributed=False, span_modalities=SPAN_MODS,
    )
    acc = defaultdict(lambda: [0.0, 0])  # modality -> [sum_ce*tok, tok]
    seen = 0
    with torch.no_grad():
        for data in loader:
            data = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in data.items()}
            _, per_mod = model(data)
            # weight each modality CE by its valid target-token count this batch
            dec_mod = data["dec_modalities"]; dec_pad = data["dec_pad_mask"]
            for mi, m in enumerate(MODALITIES):
                ntok = int(((dec_mod == mi) & dec_pad).sum().item())
                if m in per_mod and ntok > 0 and torch.isfinite(per_mod[m]).all():
                    acc[m][0] += float(per_mod[m]) * ntok
                    acc[m][1] += ntok
            seen += 1
            if seen >= n_batches: break
    ce = {m: (acc[m][0] / acc[m][1] if acc[m][1] else float("nan")) for m in MODALITIES}
    baseline = {m: math.log(VOCAB_SIZES[MOD_TO_IDX[m]]) for m in MODALITIES}
    drop = {m: baseline[m] - ce[m] for m in MODALITIES}
    print("  per-modality CE / baseln / drop:")
    for m in MODALITIES:
        print(f"    {m:16s} ce={ce[m]:.3f}  ln(V)={baseline[m]:.3f}  drop={drop[m]:.3f}")

    labels_map = {"tok_rgb@196": "RGB", "tok_audio@512": "Audio",
                  "tok_depth@196": "Depth", "tok_normal@196": "Normal",
                  "scene_desc": "Caption"}
    order = sorted(MODALITIES, key=lambda m: -drop[m])
    labels = [labels_map[m] for m in order]
    drops  = [drop[m] for m in order]
    bvals  = [baseline[m] for m in order]
    cevals = [ce[m] for m in order]

    # Two-panel: (a) absolute CE vs ln(V) baseline, (b) the drop.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    x = np.arange(len(order)); w = 0.4
    axA.bar(x - w/2, bvals, w, color=GREY["baseline"], edgecolor=GREY["edge"],
            linewidth=0.8, label=r"chance $\ln V$")
    axA.bar(x + w/2, cevals, w, color=GREY["bar"], edgecolor=GREY["edge"],
            linewidth=0.8, label="model")
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right")
    axA.set_ylabel("cross-entropy (nats)")
    axA.set_title("(a) test CE vs.\\ chance" if plt.rcParams["text.usetex"] else "(a) test CE vs. chance")
    axA.set_ylim(0, max(bvals) * 1.16)
    axA.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    for xi, c in zip(x, cevals):
        axA.text(xi + w/2, c + 0.12, f"{c:.2f}", ha="center", va="bottom", fontsize=7.5)

    bars = axB.bar(x, drops, 0.62, color=GREY["bar"], edgecolor=GREY["edge"], linewidth=0.8)
    axB.set_xticks(x); axB.set_xticklabels(labels, rotation=20, ha="right")
    axB.set_ylabel(r"CE drop $\ln V - \mathrm{CE}$ (nats)")
    axB.set_title("(b) information captured per modality")
    for b, d in zip(bars, drops):
        axB.text(b.get_x() + b.get_width()/2, d + 0.1, f"{d:.2f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axB.set_ylim(0, max(drops) * 1.18)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig1_ce_drop.{ext}")
    plt.close(fig)
    json.dump({"ce": ce, "baseline": baseline, "drop": drop,
               "n_batches": n_batches, "batch_size": batch_size},
              open(OUT_DIR / "fig1_ce_drop.json", "w"), indent=2)
    print("  saved fig1_ce_drop.{pdf,png}", flush=True)


# ============================================================================
# FIGURE 2 — depth / normal reconstruction grid
# ============================================================================
def fig2_depth_normal(n_rows=6, maskgit_iter=12, seed=0):
    print("\n[fig2] depth/normal reconstruction grid ...", flush=True)
    rng = np.random.RandomState(seed)
    # pick visually-diverse stems across distinct classes
    by_cls = defaultdict(list)
    for i in range(len(test_ds)):
        by_cls[read_class("test", test_ds.file_names[i])].append(i)
    chosen, used = [], set()
    for c in CLASSES_CANONICAL:
        if by_cls[c] and len(chosen) < n_rows:
            idx = by_cls[c][rng.randint(len(by_cls[c]))]
            chosen.append(idx); used.add(c)
    cols = ["RGB input", "GT depth", "pred depth", "GT normal", "pred normal"]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(9.2, 1.85 * len(chosen)))
    if len(chosen) == 1: axes = axes[None, :]
    for r, idx in enumerate(chosen):
        s = get_modality_tokens(test_ds, idx, aug=0)
        rgb = decode_rgb(s["tok_rgb@196"].to(DEVICE))
        gtd = decode_depth(s["tok_depth@196"].to(DEVICE))
        gtn = decode_normal(s["tok_normal@196"].to(DEVICE))
        pd_tok = maskgit_generate(
            {"tok_rgb@196": (s["tok_rgb@196"], FULL["tok_rgb@196"]),
             "scene_desc":  (s["scene_desc"],  FULL["scene_desc"])},
            "tok_depth@196", n_iter=maskgit_iter, temperature=1.0)
        pn_tok = maskgit_generate(
            {"tok_rgb@196": (s["tok_rgb@196"], FULL["tok_rgb@196"]),
             "scene_desc":  (s["scene_desc"],  FULL["scene_desc"])},
            "tok_normal@196", n_iter=maskgit_iter, temperature=1.0)
        prd = decode_depth(pd_tok); prn = decode_normal(pn_tok)
        ims = [rgb, gtd, prd, gtn, prn]
        cmaps = [None, "gray", "gray", None, None]
        for cidx, (im, cm) in enumerate(zip(ims, cmaps)):
            ax = axes[r, cidx]
            ax.imshow(im, cmap=cm); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.6)
            if r == 0: ax.set_title(cols[cidx], fontsize=10)
        axes[r, 0].set_ylabel(SHORT[s["__class__"]], fontsize=10, rotation=90, labelpad=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_depth_normal_recon.{ext}")
    plt.close(fig)
    print("  saved fig2_depth_normal_recon.{pdf,png}", flush=True)


# ============================================================================
# FIGURE 3 — caption -> RGB gallery
# ============================================================================
def fig3_caption_gallery(samples_per_class=4, maskgit_iter=12, temperature=1.0, seed=0):
    print("\n[fig3] caption->RGB gallery ...", flush=True)
    torch.manual_seed(seed)
    fig, axes = plt.subplots(len(CLASSES_CANONICAL), samples_per_class,
                             figsize=(1.7 * samples_per_class, 1.7 * len(CLASSES_CANONICAL)))
    for r, cls in enumerate(CLASSES_CANONICAL):
        tok = test_ds.text_tokenizer(cls, max_length=64, padding="max_length",
                                     truncation=True, return_tensors="pt")["input_ids"][0].long()
        for cidx in range(samples_per_class):
            rgb_tok = maskgit_generate(
                {"scene_desc": (tok, FULL["scene_desc"])},
                "tok_rgb@196", n_iter=maskgit_iter, temperature=temperature,
                temp_floor=0.7, top_p=0.9)
            img = decode_rgb(rgb_tok)
            np.save(OUT_DIR / "caption2rgb_candidates" / f"{SHORT[cls]}_{cidx}.npy", img)
            ax = axes[r, cidx]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.5)
            if cidx == 0:
                ax.set_ylabel(SHORT[cls], fontsize=10, rotation=90, labelpad=6)
    fig.suptitle("Caption $\\rightarrow$ RGB" if plt.rcParams["text.usetex"]
                 else "Caption -> RGB", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_caption2rgb_gallery.{ext}")
    plt.close(fig)
    print("  saved fig3_caption2rgb_gallery.{pdf,png}", flush=True)


# ============================================================================
# FIGURE 4 — audio -> RGB generation grid (many candidates for selection)
# ============================================================================
def fig4_audio2rgb(n_stems=8, temps=(0.8, 1.0, 1.2), seeds=(0, 1), maskgit_iter=12, seed=0):
    print("\n[fig4] audio->RGB candidates ...", flush=True)
    rng = np.random.RandomState(seed)
    by_cls = defaultdict(list)
    for i in range(len(test_ds)):
        by_cls[read_class("test", test_ds.file_names[i])].append(i)
    # prioritize the acoustically-learned classes, then fill
    priority = ["pig oinking", "sheep bleating", "dog barking", "cat meowing",
                "chicken clucking", "cow lowing", "horse neighing", "duck quacking"]
    chosen = []
    for c in priority:
        if by_cls[c] and len(chosen) < n_stems:
            chosen.append((c, by_cls[c][rng.randint(len(by_cls[c]))]))
    ncols = len(temps) * len(seeds)
    fig, axes = plt.subplots(len(chosen), ncols + 1,
                             figsize=(1.55 * (ncols + 1), 1.55 * len(chosen)))
    if len(chosen) == 1: axes = axes[None, :]
    for r, (cls, idx) in enumerate(chosen):
        s = get_modality_tokens(test_ds, idx, aug=0)
        # reference: ground-truth RGB for this clip
        gt = decode_rgb(s["tok_rgb@196"].to(DEVICE))
        axes[r, 0].imshow(gt); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.7)
        if r == 0: axes[r, 0].set_title("GT RGB", fontsize=9)
        axes[r, 0].set_ylabel(SHORT[cls], fontsize=10, rotation=90, labelpad=6)
        col = 1
        for T in temps:
            for sd in seeds:
                torch.manual_seed(1000 * sd + r)
                rgb_tok = maskgit_generate(
                    {"tok_audio@512": (s["tok_audio@512"], FULL["tok_audio@512"])},
                    "tok_rgb@196", n_iter=maskgit_iter, temperature=T,
                    temp_floor=max(0.6, 0.6 * T), top_p=0.9)
                img = decode_rgb(rgb_tok)
                np.save(OUT_DIR / "audio2rgb_candidates" / f"{SHORT[cls]}_T{T}_s{sd}.npy", img)
                ax = axes[r, col]
                ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.4)
                if r == 0:
                    ax.set_title((f"$T{{=}}{T}$" if plt.rcParams["text.usetex"] else f"T={T}")
                                 + f", s{sd}", fontsize=8)
                col += 1
    fig.suptitle("Audio $\\rightarrow$ RGB candidates" if plt.rcParams["text.usetex"]
                 else "Audio -> RGB candidates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_audio2rgb_grid.{ext}")
    plt.close(fig)
    print("  saved fig4_audio2rgb_grid.{pdf,png} + individual .npy candidates", flush=True)


# ============================================================================
# FIGURE 5 — reconstruction gallery (tokenizer round-trip, GT tokens decoded)
#   3 rows (RGB | depth | normal) x 6 columns (test samples), 2400x1200 px.
# ============================================================================
def fig5_reconstruction_gallery(n_cols=6, seed=3, px=(2400, 1200)):
    print("\n[fig5] reconstruction gallery (GT-token round-trip) ...", flush=True)
    rng = np.random.RandomState(seed)
    # one visually-distinct sample per class, first n_cols classes (shuffled)
    by_cls = defaultdict(list)
    for i in range(len(test_ds)):
        by_cls[read_class("test", test_ds.file_names[i])].append(i)
    classes = [c for c in CLASSES_CANONICAL if by_cls[c]]
    rng.shuffle(classes)
    chosen = []
    for c in classes[:n_cols]:
        chosen.append((c, by_cls[c][rng.randint(len(by_cls[c]))]))
    while len(chosen) < n_cols:  # fallback if <n_cols classes
        c = classes[len(chosen) % len(classes)]
        chosen.append((c, by_cls[c][rng.randint(len(by_cls[c]))]))

    rows = ["RGB", "Depth", "Normal"]
    # exact pixel size: figsize(inches) * dpi = px
    dpi = 200
    figsize = (px[0] / dpi, px[1] / dpi)   # (12, 6) in -> 2400x1200 @200dpi
    fig, axes = plt.subplots(3, n_cols, figsize=figsize, dpi=dpi)
    for cidx, (cls, idx) in enumerate(chosen):
        s = get_modality_tokens(test_ds, idx, aug=0)
        rgb = decode_rgb(s["tok_rgb@196"].to(DEVICE))
        dep = decode_depth(s["tok_depth@196"].to(DEVICE))
        nor = decode_normal(s["tok_normal@196"].to(DEVICE))
        for ridx, (im, cm) in enumerate([(rgb, None), (dep, "gray"), (nor, None)]):
            ax = axes[ridx, cidx]
            ax.imshow(im, cmap=cm); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.6)
            if ridx == 0:
                ax.set_title(SHORT[cls], fontsize=12)
            if cidx == 0:
                ax.set_ylabel(rows[ridx], fontsize=13, rotation=90, labelpad=8)
    fig.subplots_adjust(left=0.04, right=0.997, top=0.95, bottom=0.01,
                        wspace=0.04, hspace=0.04)
    # save at exact pixel size (no bbox='tight' which would change dimensions)
    fig.savefig(OUT_DIR / "reconstruction_gallery.png", dpi=dpi, bbox_inches=None)
    fig.savefig(OUT_DIR / "reconstruction_gallery.pdf", bbox_inches=None)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(OUT_DIR / "reconstruction_gallery.png").size
    print(f"  saved reconstruction_gallery.png  ({w}x{h} px)", flush=True)


# ============================================================================
# FIGURE 6 — sanity-check directions (SINGLE forward-pass, train-aligned)
#   Shows the model works in the expected directions when decoded the way it was
#   trained (predict all held-out tokens in ONE step, argmax) -- isolating the
#   audio-generation failure as a specific limitation, and giving direct
#   evidence for the train/inference (single-step vs MaskGIT) mismatch.
# ============================================================================
@torch.no_grad()
def single_step_predict(input_mods, target_mod, gt_tokens=None):
    """One forward pass: put ALL target positions in the decoder query, argmax
    per position. Returns (pred_tokens, token_accuracy_or_None)."""
    tv = VOCAB_SIZES[MOD_TO_IDX[target_mod]]
    pos = FULL[target_mod]
    logits = forward_logits(input_mods, {target_mod: pos})[target_mod][..., :tv]
    pred = logits.argmax(-1)
    acc = None
    if gt_tokens is not None:
        gt = gt_tokens.to(DEVICE)
        acc = float((pred == gt).float().mean().item())
    return pred, acc

def fig6_sanity_directions(n_rows=5, seed=2):
    print("\n[fig6] sanity-check directions (single forward pass) ...", flush=True)
    rng = np.random.RandomState(seed)
    by_cls = defaultdict(list)
    for i in range(len(test_ds)):
        by_cls[read_class("test", test_ds.file_names[i])].append(i)
    classes = [c for c in CLASSES_CANONICAL if by_cls[c]]
    rng.shuffle(classes)
    chosen = [(c, by_cls[c][rng.randint(len(by_cls[c]))]) for c in classes[:n_rows]]

    cols = ["RGB (input)", r"RGB$\to$Depth", "GT Depth",
            r"RGB$\to$Normal", "GT Normal", r"[D+N+cap]$\to$RGB"]
    cols_plain = ["RGB (input)", "RGB->Depth", "GT Depth",
                  "RGB->Normal", "GT Normal", "[D+N+cap]->RGB"]
    use_tex = plt.rcParams["text.usetex"]
    headers = cols if use_tex else cols_plain

    accs = defaultdict(list)
    fig, axes = plt.subplots(len(chosen), 6, figsize=(11, 1.9 * len(chosen)))
    if len(chosen) == 1: axes = axes[None, :]
    for r, (cls, idx) in enumerate(chosen):
        s = get_modality_tokens(test_ds, idx, aug=0)
        rgb_in  = decode_rgb(s["tok_rgb@196"].to(DEVICE))
        gt_dep  = decode_depth(s["tok_depth@196"].to(DEVICE))
        gt_nor  = decode_normal(s["tok_normal@196"].to(DEVICE))
        # RGB -> Depth (single step)
        pd, ad = single_step_predict(
            {"tok_rgb@196": (s["tok_rgb@196"], FULL["tok_rgb@196"]),
             "scene_desc":  (s["scene_desc"],  FULL["scene_desc"])},
            "tok_depth@196", gt_tokens=s["tok_depth@196"])
        # RGB -> Normal (single step)
        pn, an = single_step_predict(
            {"tok_rgb@196": (s["tok_rgb@196"], FULL["tok_rgb@196"]),
             "scene_desc":  (s["scene_desc"],  FULL["scene_desc"])},
            "tok_normal@196", gt_tokens=s["tok_normal@196"])
        # [Depth+Normal+caption] -> RGB self-reconstruction (single step)
        pr, ar = single_step_predict(
            {"tok_depth@196":  (s["tok_depth@196"],  FULL["tok_depth@196"]),
             "tok_normal@196": (s["tok_normal@196"], FULL["tok_normal@196"]),
             "scene_desc":     (s["scene_desc"],     FULL["scene_desc"])},
            "tok_rgb@196", gt_tokens=s["tok_rgb@196"])
        accs["RGB->Depth"].append(ad); accs["RGB->Normal"].append(an)
        accs["[D+N+cap]->RGB"].append(ar)
        ims   = [rgb_in, decode_depth(pd), gt_dep, decode_normal(pn), gt_nor, decode_rgb(pr)]
        cmaps = [None, "gray", "gray", None, None, None]
        for cidx, (im, cm) in enumerate(zip(ims, cmaps)):
            ax = axes[r, cidx]
            ax.imshow(im, cmap=cm); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#000000"); sp.set_linewidth(0.6)
            if r == 0: ax.set_title(headers[cidx], fontsize=10)
        axes[r, 0].set_ylabel(SHORT[cls], fontsize=10, rotation=90, labelpad=6)
    mad = np.mean([a for a in accs["RGB->Depth"] if a is not None])
    man = np.mean([a for a in accs["RGB->Normal"] if a is not None])
    mar = np.mean([a for a in accs["[D+N+cap]->RGB"] if a is not None])
    sub = (f"single forward pass (train-aligned)  --  token acc: "
           f"RGB$\\to$Depth {mad:.0%}, RGB$\\to$Normal {man:.0%}, "
           f"[D+N+cap]$\\to$RGB {mar:.0%}") if use_tex else \
          (f"single forward pass (train-aligned)  --  token acc: "
           f"RGB->Depth {mad:.0%}, RGB->Normal {man:.0%}, [D+N+cap]->RGB {mar:.0%}")
    fig.suptitle("Sanity check: conditional reconstruction in expected directions\n" + sub,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"sanity_check_directions.{ext}")
    plt.close(fig)
    json.dump({"mean_token_acc": {"RGB->Depth": float(mad), "RGB->Normal": float(man),
                                  "[D+N+cap]->RGB": float(mar)},
               "per_sample": {k: v for k, v in accs.items()}},
              open(OUT_DIR / "sanity_check_directions.json", "w"), indent=2)
    print(f"  token acc: RGB->Depth {mad:.3f}  RGB->Normal {man:.3f}  [D+N+cap]->RGB {mar:.3f}")
    print("  saved sanity_check_directions.{pdf,png}", flush=True)


# ============================================================================
# FIGURE 7 — tokenizer fidelity: is the blur the source or the tokenizer?
#   Round-trips (a) real source frames and (b) a synthetic SHARP test pattern
#   through the RGB DiVAE (encode->decode). If even the sharp pattern comes back
#   blocky, the 196-token bottleneck is the ceiling, not source quality.
# ============================================================================
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

RAW_SEARCH_DIRS = [
    "/path/to/nano4M-Audio/data/raw_v4_merged",
    "/path/to/nano4M-Audio/data/_data2_staging/data",
    "/path/to/nano4M-Audio/data/_lionsheep_staging/extracted",
]

def _find_raw_frame(stem):
    import glob
    for base in RAW_SEARCH_DIRS:
        hits = glob.glob(f"{base}/*/*/{stem}/k00.jpg") + \
               glob.glob(f"{base}/*/*/{stem}/frame.jpg")
        if hits:
            return hits[0]
    return None

def _preprocess_rgb(pil):
    img = pil.convert("RGB")
    w, h = img.size; s = 224 / min(w, h)
    img = img.resize((max(224, int(round(w*s))), max(224, int(round(h*s)))), Image.BICUBIC)
    img = TF.center_crop(img, [224, 224])
    t = TF.to_tensor(img) * 2 - 1
    return t  # [3,224,224] in [-1,1]

@torch.no_grad()
def _roundtrip_rgb(t_in):
    """[3,224,224] in [-1,1] -> tokenize -> decode -> [-1,1]."""
    out = rgb_detok.tokenize(t_in.unsqueeze(0).to(DEVICE))
    toks = out[0] if isinstance(out, tuple) else out
    toks = toks.reshape(1, -1).long()
    if toks.shape[1] != 196:
        toks = toks.reshape(1, 196)
    rec = rgb_detok.decode_tokens(toks.view(1, 14, 14))[0].clamp(-1, 1)
    return rec.cpu()

def _to01(t): return (t.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

def _match_size(a, b):
    """The 4M-16k RGB DiVAE decodes at 448x448 even from a 224 input. Resize the
    smaller HxWx3 array up to the larger so metrics compare like-for-like content."""
    if a.shape[:2] == b.shape[:2]:
        return a, b
    H = max(a.shape[0], b.shape[0]); W = max(a.shape[1], b.shape[1])
    def up(x):
        if x.shape[:2] == (H, W):
            return x
        return np.asarray(Image.fromarray((x * 255).astype(np.uint8))
                          .resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
    return up(a), up(b)

def _psnr(a, b):  # a,b in [0,1] HxWx3
    a, b = _match_size(a, b)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)

def _ssim_gray(a, b):  # simple global SSIM on luminance
    a, b = _match_size(a, b)
    ag = a.mean(2); bg = b.mean(2)
    mu_a, mu_b = ag.mean(), bg.mean()
    va, vb = ag.var(), bg.var()
    cov = ((ag - mu_a) * (bg - mu_b)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(((2*mu_a*mu_b + c1)*(2*cov + c2)) /
                 ((mu_a**2 + mu_b**2 + c1)*(va + vb + c2)))

def _synthetic_sharp():
    """A 224x224 high-frequency test image: checkerboard + text + edges."""
    im = Image.new("RGB", (224, 224), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(0, 224, 8):           # fine checkerboard (8px squares)
        for j in range(0, 224, 8):
            if (i//8 + j//8) % 2 == 0:
                d.rectangle([i, j, i+8, j+8], fill=(0, 0, 0))
    d.rectangle([20, 90, 204, 134], fill=(255, 255, 255))
    d.text((30, 100), "SHARP 4M-16k", fill=(200, 0, 0))
    for k, x in enumerate(range(30, 200, 6)):   # thin vertical lines
        d.line([x, 150, x, 210], fill=(0, 0, 255), width=1)
    return TF.to_tensor(im) * 2 - 1

def fig7_tokenizer_fidelity(n_real=4, n_quant=60, seed=5):
    print("\n[fig7] tokenizer fidelity (round-trip) ...", flush=True)
    rng = np.random.RandomState(seed)
    # quantitative: mean PSNR/SSIM over n_quant real frames we can locate
    psnrs, ssims, n_used = [], [], 0
    order = list(range(len(test_ds))); rng.shuffle(order)
    real_examples = []
    for idx in order:
        stem = test_ds.file_names[idx]
        raw = _find_raw_frame(stem)
        if raw is None:
            continue
        t_in = _preprocess_rgb(Image.open(raw))
        rec = _roundtrip_rgb(t_in)
        a, b = _to01(t_in), _to01(rec)
        psnrs.append(_psnr(a, b)); ssims.append(_ssim_gray(a, b)); n_used += 1
        if len(real_examples) < n_real:
            real_examples.append((read_class("test", stem), a, b,
                                  _psnr(a, b), _ssim_gray(a, b)))
        if n_used >= n_quant:
            break
    mP, mS = (float(np.mean(psnrs)) if psnrs else float("nan"),
              float(np.mean(ssims)) if ssims else float("nan"))
    # synthetic sharp round-trip
    syn_in = _synthetic_sharp(); syn_rec = _roundtrip_rgb(syn_in)
    sa, sb = _to01(syn_in), _to01(syn_rec)
    sP, sS = _psnr(sa, sb), _ssim_gray(sa, sb)
    comp = (224*224*3*8) / (196*14)   # bits raw / bits tokens
    print(f"  real round-trip (n={n_used}):  PSNR={mP:.2f} dB  SSIM={mS:.3f}")
    print(f"  synthetic sharp round-trip:    PSNR={sP:.2f} dB  SSIM={sS:.3f}")
    print(f"  compression ratio ~ {comp:.0f}x  (196 tokens x 14 bits = {196*14} bits/image)")

    nrows = len(real_examples) + 1
    fig, axes = plt.subplots(nrows, 2, figsize=(4.2, 2.05 * nrows))
    if nrows == 1: axes = axes[None, :]
    axes[0, 0].set_title("source (224 crop)", fontsize=10)
    axes[0, 1].set_title("tokenizer round-trip", fontsize=10)
    # synthetic first
    axes[0, 0].imshow(sa); axes[0, 1].imshow(sb)
    axes[0, 0].set_ylabel("SHARP\\,test" if plt.rcParams["text.usetex"] else "SHARP test",
                          fontsize=9, rotation=90, labelpad=6)
    axes[0, 1].text(0.5, -0.13, f"PSNR {sP:.1f} dB  SSIM {sS:.2f}",
                    ha="center", va="top", transform=axes[0, 1].transAxes, fontsize=8)
    for r, (cls, a, b, p, s) in enumerate(real_examples, start=1):
        axes[r, 0].imshow(a); axes[r, 1].imshow(b)
        axes[r, 0].set_ylabel(SHORT[cls], fontsize=9, rotation=90, labelpad=6)
        axes[r, 1].text(0.5, -0.13, f"PSNR {p:.1f} dB  SSIM {s:.2f}",
                        ha="center", va="top", transform=axes[r, 1].transAxes, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#000"); sp.set_linewidth(0.6)
    ttl = (f"RGB tokenizer fidelity: 196 tokens/image ($\\sim${comp:.0f}$\\times$ compression)\n"
           f"real frames PSNR {mP:.1f} dB / SSIM {mS:.2f} (n={n_used}); "
           f"sharp pattern PSNR {sP:.1f} dB / SSIM {sS:.2f}") if plt.rcParams["text.usetex"] else \
          (f"RGB tokenizer fidelity: 196 tokens/image (~{comp:.0f}x compression)\n"
           f"real PSNR {mP:.1f}dB/SSIM {mS:.2f} (n={n_used}); sharp PSNR {sP:.1f}dB/SSIM {sS:.2f}")
    fig.suptitle(ttl, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig7_tokenizer_fidelity.{ext}")
    plt.close(fig)
    json.dump({"real_mean_psnr": mP, "real_mean_ssim": mS, "n_real": n_used,
               "synthetic_psnr": sP, "synthetic_ssim": sS,
               "tokens_per_image": 196, "bits_per_token": 14,
               "compression_ratio_vs_uint8": comp},
              open(OUT_DIR / "fig7_tokenizer_fidelity.json", "w"), indent=2)
    print("  saved fig7_tokenizer_fidelity.{pdf,png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["1", "2", "3", "4", "5", "6", "7"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny params to validate all code paths fast")
    args = ap.parse_args()
    setup_style()
    if args.smoke:
        runners = {
            "1": lambda: fig1_ce_drop(n_batches=2, batch_size=16),
            "2": lambda: fig2_depth_normal(n_rows=2, maskgit_iter=4),
            "3": lambda: fig3_caption_gallery(samples_per_class=2, maskgit_iter=4),
            "4": lambda: fig4_audio2rgb(n_stems=2, temps=(1.0,), seeds=(0,), maskgit_iter=4),
        }
    else:
        runners = {"1": fig1_ce_drop, "2": fig2_depth_normal,
                   "3": fig3_caption_gallery, "4": fig4_audio2rgb,
                   "5": fig5_reconstruction_gallery, "6": fig6_sanity_directions,
                   "7": fig7_tokenizer_fidelity}
    for k in args.only:
        try:
            runners[k]()
        except Exception as e:
            print(f"[fig{k}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nALL FIGURES IN: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
