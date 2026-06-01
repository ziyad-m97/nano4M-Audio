#!/usr/bin/env python
"""
AudioSet -> YouTube pigeon/dove pair extractor.

Pipeline per clip:
  1. yt-dlp section download [start, end] (~10s @ <=480p mp4)
  2. extract 10 JPEG frames evenly
  3. CLIP gate: max cosine vs "a photo of a pigeon or dove" >= 0.25
  4. extract full_10s.wav (24kHz mono PCM16)
  5. find 5 RMS peaks -> 5 candidate windows of `--duration` s
  6. drop windows with any voiced speech (Silero VAD)
  7. pick surviving window with max PANN score (index 115 = "Pigeon, dove")
  8. require PANN >= 0.30
  9. save audio.wav + frame.jpg + k00..k09.jpg + scores.txt

Status tokens (one per attempt) appear with leading/trailing space so user's
`grep -c " OK "` etc. counts work as documented.
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import soundfile as sf
import librosa
import torch
from PIL import Image
import imageio_ffmpeg

# Force UTF-8 on stdout/stderr (Windows console defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------- constants (class-specific defaults; overridable via CLI) ----------
PIGEON_MID = "/m/0h0rv"
PANN_PIGEON_IDX = 115
DEFAULT_CLIP_PROMPT = "a photo of a pigeon or dove"
TARGET_SR = 24000          # output audio sample rate (mono PCM16)
PANN_SR = 32000            # PANN expected sample rate
VAD_SR = 16000             # Silero VAD expected sample rate
N_FRAMES = 10
N_AUDIO_CANDIDATES = 5
MIN_PEAK_SEPARATION_S = 0.5

# Prefer a "ffmpeg.exe" sitting in ./bin so yt-dlp can find it by canonical name.
_BIN_DIR = Path(__file__).resolve().parent / "bin"
_BIN_FFMPEG = _BIN_DIR / "ffmpeg.exe"
if _BIN_FFMPEG.exists():
    FFMPEG_BIN = str(_BIN_FFMPEG)
    FFMPEG_DIR = str(_BIN_DIR)
else:
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    FFMPEG_DIR = str(Path(FFMPEG_BIN).parent)
# Also expose on PATH for any child that probes via PATH.
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ---------- argparse ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audioset_meta", required=True, type=Path)
    p.add_argument("--labels_csv", required=True, type=Path)  # parsed but not strictly needed
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--duration", type=float, default=256.0 / 75.0)  # 3.41333...
    p.add_argument("--pann_threshold", type=float, default=0.30)
    p.add_argument("--clip_threshold", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--cookies", type=str, default=os.environ.get("YT_DLP_COOKIES", ""))
    p.add_argument("--sleep_requests", type=float, default=float(os.environ.get("YT_SLEEP_REQUESTS", 3)))
    p.add_argument("--sleep_min", type=float, default=float(os.environ.get("YT_SLEEP_MIN", 2)))
    p.add_argument("--sleep_max", type=float, default=float(os.environ.get("YT_SLEEP_MAX", 6)))
    p.add_argument("--yt_retries", type=int, default=int(os.environ.get("YT_RETRIES", 5)))
    # Class-specific overrides (defaults = pigeon)
    p.add_argument("--class_mid", type=str, default=PIGEON_MID,
                   help="AudioSet MID to filter (e.g. /m/0cdnk for lions, /m/07bgp for sheep)")
    p.add_argument("--pann_idx", type=int, default=PANN_PIGEON_IDX,
                   help="PANN class index for the target class (e.g. 109 lion, 97 sheep)")
    p.add_argument("--clip_prompt", type=str, default=DEFAULT_CLIP_PROMPT,
                   help="CLIP text prompt for visual gate")
    p.add_argument("--skip_csv", type=str, default="",
                   help="CSV with a 'ytid' column; rows already collected, skip them")
    return p.parse_args()


# ---------- audioset CSV parsing ----------
def parse_audioset_csv(path, split_name, target_mid):
    """Yield dict(ytid, start, end, labels_set, split) for rows matching target_mid."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                # Format: YTID, start, end, "label1,label2,..."  (spaces present)
                first_q = line.index('"')
                left = line[:first_q].rstrip(", \t")
                right = line[first_q + 1:].rstrip().rstrip('"')
                fields = [x.strip() for x in left.split(",")]
                if len(fields) < 3:
                    continue
                ytid, start_s, end_s = fields[0], fields[1], fields[2]
                labels = set(l.strip() for l in right.split(","))
            except Exception:
                continue
            if target_mid not in labels:
                continue
            try:
                start = float(start_s)
                end = float(end_s)
            except ValueError:
                continue
            out.append({
                "ytid": ytid, "start": start, "end": end,
                "labels": labels, "split": split_name,
            })
    return out


def load_skip_ytids(csv_path):
    """Load a set of ytids to skip from a CSV with a 'ytid' column."""
    if not csv_path:
        return set()
    p = Path(csv_path)
    if not p.exists():
        print(f"[WARN] skip_csv not found: {csv_path}", flush=True)
        return set()
    ytids = set()
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "ytid" not in (reader.fieldnames or []):
            print(f"[WARN] skip_csv has no 'ytid' column. Cols: {reader.fieldnames}", flush=True)
            return set()
        for row in reader:
            yt = (row.get("ytid") or "").strip()
            if yt:
                ytids.add(yt)
    return ytids


def build_pool(meta_dir, seed, target_mid=PIGEON_MID, skip_ytids=None):
    paths = {
        "train": [
            meta_dir / "balanced_train_segments.csv",
            meta_dir / "unbalanced_train_segments.csv",
        ],
        "test": [meta_dir / "eval_segments.csv"],
    }
    skip_ytids = skip_ytids or set()
    cands = []
    for split, files in paths.items():
        for f in files:
            if not f.exists():
                print(f"[WARN] missing {f}", flush=True)
                continue
            cands.extend(parse_audioset_csv(f, split, target_mid))
    # dedupe by ytid (1 video max). Prefer eval (test) over train if collision.
    by_id = {}
    for c in cands:
        if c["ytid"] in by_id:
            if c["split"] == "test" and by_id[c["ytid"]]["split"] != "test":
                by_id[c["ytid"]] = c
            continue
        by_id[c["ytid"]] = c
    # Drop already-collected ytids
    before = len(by_id)
    for yt in list(by_id.keys()):
        if yt in skip_ytids:
            del by_id[yt]
    skipped = before - len(by_id)
    uniq = list(by_id.values())
    rng = random.Random(seed)
    rng.shuffle(uniq)
    if skipped:
        print(f"[pool] {skipped} ytids skipped via --skip_csv ({len(uniq)} remaining of {before})", flush=True)
    return uniq


# ---------- yt-dlp download ----------
def yt_download(ytid, start, end, dst_mp4, cookies, retries, sleep_req, sleep_min, sleep_max):
    """Download [start, end] segment of the YouTube video. Returns (ok, message)."""
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp = dst_mp4.with_suffix(".part.mp4")
    if tmp.exists():
        try: tmp.unlink()
        except: pass

    ydl_opts = {
        # 18/22 = legacy combined mp4; else fall back to any <=480p combined; else merge separate streams.
        "format": "18/22/best[height<=480][acodec!=none]/best[acodec!=none]/bestvideo[height<=480]+bestaudio/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(tmp),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": FFMPEG_DIR,
        "retries": retries,
        "fragment_retries": retries,
        "extractor_retries": retries,
        "concurrent_fragment_downloads": 1,
        "sleep_interval_requests": sleep_req,
        "sleep_interval": sleep_min,
        "max_sleep_interval": sleep_max,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(float(start), float(end))]),
        "force_keyframes_at_cuts": True,
        "overwrites": True,
    }
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        msg = str(e).splitlines()[-1][:160] if str(e) else type(e).__name__
        return False, f"yt_dlp:{msg}"
    if not tmp.exists() or tmp.stat().st_size < 1024:
        return False, "yt_dlp:empty"
    tmp.rename(dst_mp4)
    return True, "ok"


# ---------- ffmpeg helpers ----------
def run_ffmpeg(args, timeout=60):
    """Run ffmpeg with given args (list). Returns (ok, stderr_tail)."""
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"] + args
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if r.returncode != 0:
        return False, r.stderr.decode("utf-8", "ignore")[-200:]
    return True, ""


def ffprobe_duration(path):
    """Return duration in seconds (float) or None."""
    # ffprobe is bundled alongside ffmpeg in imageio binary? imageio-ffmpeg only has ffmpeg.
    # Fallback: use ffmpeg -i then parse stderr.
    cmd = [FFMPEG_BIN, "-i", str(path)]
    r = subprocess.run(cmd, capture_output=True)
    err = r.stderr.decode("utf-8", "ignore")
    for line in err.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            try:
                t = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = t.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                pass
    return None


def extract_frames(mp4_path, out_dir, n=N_FRAMES):
    """Extract n evenly-spaced JPEG frames to out_dir/k00.jpg..k(n-1).jpg.
    Returns (list_of_paths, duration_seconds) or (None, None)."""
    dur = ffprobe_duration(mp4_path)
    if not dur or dur < 0.5:
        return None, None
    out_paths = []
    for i in range(n):
        t = (i + 0.5) / n * dur
        out_p = out_dir / f"k{i:02d}.jpg"
        ok, err = run_ffmpeg([
            "-ss", f"{t:.3f}", "-i", str(mp4_path),
            "-frames:v", "1", "-q:v", "3",
            "-vf", "scale='min(640,iw)':-2",
            str(out_p),
        ])
        if not ok or not out_p.exists():
            # tolerate occasional missing frame
            continue
        out_paths.append(out_p)
    if len(out_paths) < max(3, n // 2):
        return None, dur
    return out_paths, dur


def extract_full_audio_wav(mp4_path, wav_path):
    """Decode audio of mp4 to 24kHz mono PCM16 wav."""
    ok, err = run_ffmpeg([
        "-i", str(mp4_path),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le",
        str(wav_path),
    ])
    return ok, err


# ---------- model holders ----------
_MODELS = {}
_MODEL_LOCK = threading.Lock()

def get_models(clip_prompt=DEFAULT_CLIP_PROMPT):
    if "ready" in _MODELS and _MODELS.get("clip_prompt") == clip_prompt:
        return _MODELS
    with _MODEL_LOCK:
        if "ready" in _MODELS and _MODELS.get("clip_prompt") == clip_prompt:
            return _MODELS
        first = "ready" not in _MODELS
        if first:
            print("[init] loading models on CPU...", flush=True)
        import open_clip
        from silero_vad import load_silero_vad, get_speech_timestamps
        from panns_inference import AudioTagging

        device = torch.device("cpu")
        if first:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

            clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            clip_model.eval().to(device)
            clip_tok = open_clip.get_tokenizer("ViT-B-32")
        else:
            clip_model = _MODELS["clip_model"]
            clip_pre = _MODELS["clip_pre"]
            clip_tok = open_clip.get_tokenizer("ViT-B-32")

        with torch.no_grad():
            tok = clip_tok([clip_prompt]).to(device)
            text_emb = clip_model.encode_text(tok)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        if first:
            vad = load_silero_vad()
            at = AudioTagging(checkpoint_path=None, device="cpu")
            _MODELS.update({
                "clip_model": clip_model,
                "clip_pre": clip_pre,
                "vad": vad,
                "vad_fn": get_speech_timestamps,
                "panns": at,
                "device": device,
                "ready": True,
            })

        _MODELS["clip_text_emb"] = text_emb
        _MODELS["clip_prompt"] = clip_prompt
        if first:
            print("[init] models ready", flush=True)
        else:
            print(f"[init] CLIP prompt updated to: {clip_prompt!r}", flush=True)
        return _MODELS


# ---------- per-step scoring ----------
def clip_scores(frame_paths):
    """Return list[float] cosine similarity per frame, in input order."""
    m = get_models()
    images = []
    for p in frame_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(m["clip_pre"](img))
        except Exception:
            images.append(None)
    valid = [(i, im) for i, im in enumerate(images) if im is not None]
    if not valid:
        return [0.0] * len(frame_paths)
    batch = torch.stack([im for _, im in valid]).to(m["device"])
    with torch.no_grad(), _MODEL_LOCK:
        emb = m["clip_model"].encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb @ m["clip_text_emb"].T).squeeze(-1).cpu().numpy()
    scores = [0.0] * len(frame_paths)
    for (i, _), s in zip(valid, sims):
        scores[i] = float(s)
    return scores


def vad_voiced_fraction(wav_24k, sr=TARGET_SR):
    """Return fraction (0..1) of speech samples in waveform."""
    if len(wav_24k) == 0:
        return 0.0
    m = get_models()
    if sr != VAD_SR:
        wav_16k = librosa.resample(wav_24k.astype(np.float32), orig_sr=sr, target_sr=VAD_SR)
    else:
        wav_16k = wav_24k.astype(np.float32)
    t = torch.from_numpy(wav_16k)
    with torch.no_grad(), _MODEL_LOCK:
        ts = m["vad_fn"](t, m["vad"], sampling_rate=VAD_SR, return_seconds=False)
    if not ts:
        return 0.0
    voiced = sum(seg["end"] - seg["start"] for seg in ts)
    return float(voiced) / float(len(wav_16k))


def pann_class_score(wav_24k, sr=TARGET_SR, pann_idx=PANN_PIGEON_IDX):
    m = get_models()
    if sr != PANN_SR:
        wav_32k = librosa.resample(wav_24k.astype(np.float32), orig_sr=sr, target_sr=PANN_SR)
    else:
        wav_32k = wav_24k.astype(np.float32)
    x = wav_32k[None, :]
    with torch.no_grad(), _MODEL_LOCK:
        clipwise, _ = m["panns"].inference(x)
    return float(clipwise[0, pann_idx])


# Backwards-compat alias kept in case anything imports the old name
pann_pigeon_score = pann_class_score


def find_rms_peaks(wav, sr, n_peaks=N_AUDIO_CANDIDATES, min_sep_s=MIN_PEAK_SEPARATION_S):
    """Return list of peak times (seconds), at most n_peaks, separated by min_sep_s."""
    hop = 256
    frame = 1024
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    order = np.argsort(rms)[::-1]
    picked_t = []
    for idx in order:
        t = float(times[idx])
        if all(abs(t - pt) >= min_sep_s for pt in picked_t):
            picked_t.append(t)
            if len(picked_t) >= n_peaks:
                break
    return picked_t


def slice_window(wav, sr, center_t, dur_s):
    """Return (segment, t_start, t_end). Clipped to bounds."""
    half = dur_s / 2.0
    t0 = max(0.0, center_t - half)
    t1 = t0 + dur_s
    if t1 > len(wav) / sr:
        t1 = len(wav) / sr
        t0 = max(0.0, t1 - dur_s)
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))
    target_n = int(round(dur_s * sr))
    seg = wav[i0:i0 + target_n]
    # pad with zeros if short (rare)
    if len(seg) < target_n:
        seg = np.pad(seg, (0, target_n - len(seg)))
    return seg, i0 / sr, (i0 + target_n) / sr


# ---------- per-candidate worker ----------
class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"OK": 0, "FAIL_DL": 0, "FAIL_CLIP": 0, "FAIL_VAD": 0,
                  "FAIL_PANN": 0, "FAIL_OTHER": 0, "attempted": 0,
                  "train": 0, "test": 0, "skipped_existing": 0}
    def inc(self, k, n=1):
        with self.lock:
            self.d[k] = self.d.get(k, 0) + n
    def snapshot(self):
        with self.lock:
            return dict(self.d)


def process_one(c, args, counters, failed_writer, failed_lock):
    """Process one AudioSet candidate. Returns status string."""
    ytid = c["ytid"]; start = c["start"]; end = c["end"]; split = c["split"]
    stem = f"as_{ytid}_{int(round(start)):06d}"
    sample_dir = args.out_dir / split / stem
    audio_target = sample_dir / "audio.wav"
    if audio_target.exists():
        counters.inc("skipped_existing")
        return f"{stem}  SKIP existing"

    counters.inc("attempted")
    sample_dir.mkdir(parents=True, exist_ok=True)
    mp4 = sample_dir / "_src.mp4"
    full_wav = sample_dir / "full_10s.wav"
    scores_txt = sample_dir / "scores.txt"

    def cleanup_partial():
        # Remove sample_dir if we failed before having an audio.wav (keep nothing).
        try:
            if sample_dir.exists() and not audio_target.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
        except Exception:
            pass

    # 1) download
    ok, msg = yt_download(ytid, start, end, mp4, args.cookies,
                          args.yt_retries, args.sleep_requests,
                          args.sleep_min, args.sleep_max)
    if not ok:
        counters.inc("FAIL_DL")
        with failed_lock:
            failed_writer.writerow([ytid, split, "DL", msg])
        cleanup_partial()
        return f"{stem}  FAIL_DL {msg}"

    try:
        # 2) extract frames
        frame_paths, video_dur = extract_frames(mp4, sample_dir, n=N_FRAMES)
        if frame_paths is None:
            counters.inc("FAIL_OTHER")
            with failed_lock:
                failed_writer.writerow([ytid, split, "FRAMES", f"dur={video_dur}"])
            cleanup_partial()
            return f"{stem}  FAIL_OTHER frames"

        # 3) CLIP gate
        c_scores = clip_scores(frame_paths)
        best_idx = int(np.argmax(c_scores))
        best_clip = float(c_scores[best_idx])
        if best_clip < args.clip_threshold:
            counters.inc("FAIL_CLIP")
            with failed_lock:
                failed_writer.writerow([ytid, split, "CLIP", f"{best_clip:.3f}"])
            cleanup_partial()
            return f"{stem}  FAIL_CLIP best={best_clip:.3f}"

        # 4) full audio
        ok, err = extract_full_audio_wav(mp4, full_wav)
        if not ok or not full_wav.exists():
            counters.inc("FAIL_OTHER")
            with failed_lock:
                failed_writer.writerow([ytid, split, "AUDIO", err[:160]])
            cleanup_partial()
            return f"{stem}  FAIL_OTHER audio:{err[:80]}"

        wav, sr = sf.read(str(full_wav), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        if len(wav) < int(args.duration * sr):
            counters.inc("FAIL_OTHER")
            with failed_lock:
                failed_writer.writerow([ytid, split, "AUDIO_SHORT", f"len={len(wav)}"])
            cleanup_partial()
            return f"{stem}  FAIL_OTHER too_short"

        # 5+6) RMS peaks -> candidate windows -> VAD filter
        peaks = find_rms_peaks(wav, sr)
        if not peaks:
            counters.inc("FAIL_OTHER")
            cleanup_partial()
            return f"{stem}  FAIL_OTHER no_peaks"

        cands = []
        for pt in peaks:
            seg, t0, t1 = slice_window(wav, sr, pt, args.duration)
            vf = vad_voiced_fraction(seg, sr=sr)
            cands.append({"t0": t0, "t1": t1, "seg": seg, "vad": vf})

        unvoiced = [c for c in cands if c["vad"] == 0.0]
        if not unvoiced:
            counters.inc("FAIL_VAD")
            min_vad = min(c["vad"] for c in cands)
            with failed_lock:
                failed_writer.writerow([ytid, split, "VAD", f"min_vad={min_vad:.3f}"])
            cleanup_partial()
            return f"{stem}  FAIL_VAD min={min_vad:.3f}"

        # 7+8) PANN per unvoiced candidate
        best = None
        best_score = -1.0
        for c in unvoiced:
            s = pann_class_score(c["seg"], sr=sr, pann_idx=args.pann_idx)
            c["pann"] = s
            if s > best_score:
                best_score = s; best = c
        if best_score < args.pann_threshold:
            counters.inc("FAIL_PANN")
            with failed_lock:
                failed_writer.writerow([ytid, split, "PANN", f"{best_score:.3f}"])
            cleanup_partial()
            return f"{stem}  FAIL_PANN max={best_score:.3f}"

        # 9) save
        sf.write(str(audio_target), best["seg"], sr, subtype="PCM_16")

        # canonical "frame.jpg" = best CLIP frame
        try:
            shutil.copyfile(frame_paths[best_idx], sample_dir / "frame.jpg")
        except Exception:
            pass

        with open(scores_txt, "w", encoding="utf-8") as f:
            f.write(f"ytid\t{ytid}\n")
            f.write(f"split\t{split}\n")
            f.write(f"yt_start\t{start:.3f}\n")
            f.write(f"yt_end\t{end:.3f}\n")
            f.write(f"video_duration\t{video_dur:.3f}\n")
            f.write(f"clip_best_idx\t{best_idx}\n")
            f.write(f"clip_best_score\t{best_clip:.4f}\n")
            f.write("clip_scores\t" + ",".join(f"{s:.4f}" for s in c_scores) + "\n")
            f.write(f"audio_window_t0\t{best['t0']:.3f}\n")
            f.write(f"audio_window_t1\t{best['t1']:.3f}\n")
            f.write(f"audio_duration\t{args.duration:.6f}\n")
            f.write(f"audio_sample_rate\t{sr}\n")
            f.write(f"pann_score\t{best_score:.4f}\n")
            f.write(f"pann_class_idx\t{args.pann_idx}\n")
            try:
                import panns_inference as _pi
                pann_label = _pi.config.labels[args.pann_idx]
            except Exception:
                pann_label = ""
            f.write(f"pann_class\t{pann_label}\n")
            f.write(f"audioset_mid\t{args.class_mid}\n")
            f.write(f"clip_prompt\t{args.clip_prompt}\n")
            f.write(f"vad_voiced_fraction\t0.0\n")
            f.write("candidates:\n")
            for c in cands:
                f.write(f"\twin=[{c['t0']:.3f},{c['t1']:.3f}] vad={c['vad']:.4f} "
                        f"pann={c.get('pann', float('nan')):.4f}\n")

        # remove the heavy mp4 to save disk
        try: mp4.unlink()
        except Exception: pass

        counters.inc("OK")
        counters.inc(split)
        return f"{stem}  OK pann={best_score:.3f} clip={best_clip:.3f}"

    except Exception as e:
        counters.inc("FAIL_OTHER")
        tb = traceback.format_exc(limit=2).strip().splitlines()[-1][:160]
        with failed_lock:
            failed_writer.writerow([ytid, split, "EXC", tb])
        cleanup_partial()
        return f"{stem}  FAIL_OTHER exc:{tb[:80]}"


# ---------- main ----------
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train").mkdir(exist_ok=True)
    (args.out_dir / "test").mkdir(exist_ok=True)

    print(f"[main] seed={args.seed} workers={args.workers} target_n={args.n}", flush=True)
    print(f"[main] ffmpeg={FFMPEG_BIN}", flush=True)
    print(f"[main] cookies={args.cookies or '<none>'}", flush=True)

    print(f"[main] target class_mid={args.class_mid} pann_idx={args.pann_idx}", flush=True)
    print(f"[main] clip_prompt={args.clip_prompt!r}", flush=True)
    skip_ytids = load_skip_ytids(args.skip_csv) if args.skip_csv else set()
    if args.skip_csv:
        print(f"[main] skip_csv={args.skip_csv} ({len(skip_ytids)} ytids loaded)", flush=True)
    pool = build_pool(args.audioset_meta, args.seed, target_mid=args.class_mid, skip_ytids=skip_ytids)
    print(f"[main] pool_size={len(pool)} (after dedupe + shuffle)", flush=True)

    counters = Counters()
    failed_path = args.out_dir / "failed.csv"
    new_failed = not failed_path.exists()
    failed_f = open(failed_path, "a", newline="", encoding="utf-8")
    failed_writer = csv.writer(failed_f)
    failed_lock = threading.Lock()
    if new_failed:
        failed_writer.writerow(["ytid", "split", "stage", "msg"])
        failed_f.flush()

    # warm models in main thread (and lock the CLIP text embedding to this run's prompt)
    get_models(clip_prompt=args.clip_prompt)

    t0 = time.time()
    targeted = min(args.n, len(pool))

    pending = list(pool)
    idx = 0
    in_flight = {}

    def submit_next(ex):
        nonlocal idx
        while idx < len(pending) and counters.snapshot()["OK"] < args.n:
            c = pending[idx]
            idx += 1
            fut = ex.submit(process_one, c, args, counters, failed_writer, failed_lock)
            in_flight[fut] = c
            return True
        return False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        # prime
        for _ in range(args.workers):
            submit_next(ex)

        while in_flight:
            done = next(as_completed(in_flight))
            c = in_flight.pop(done)
            try:
                line = done.result()
            except Exception as e:
                line = f"as_{c['ytid']}_xxx  FAIL_OTHER exc:{e}"
                counters.inc("FAIL_OTHER")
            print(line, flush=True)
            failed_f.flush()
            snap = counters.snapshot()
            if snap["OK"] >= args.n:
                # don't submit more; drain remaining
                continue
            submit_next(ex)

    failed_f.close()
    snap = counters.snapshot()
    snap["elapsed_sec"] = round(time.time() - t0, 1)
    snap["pool_size"] = len(pool)
    snap["target_n"] = args.n
    snap["seed"] = args.seed
    with open(args.out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    print(f"[main] DONE {snap}", flush=True)


if __name__ == "__main__":
    main()
