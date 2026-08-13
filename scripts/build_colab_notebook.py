"""Generate the CORAAL whole-recording oracle notebook (Colab or Kaggle).

Installs the classroom_asr package from GitHub. Scores at the **whole-recording**
level: every branch transcribes the entire interview (no reference segment
boundaries fed to any model), and WER / candidate-oracle are computed over the
full transcript. Uses all GPUs (e.g. Kaggle 2x T4).

Run:  python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

GITHUB_REPO = "Telov/classroom-asr"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(r"""
# CORAAL whole-recording candidate-oracle (Colab **or** Kaggle)

Runs the candidate-oracle gate (§18.2) on ~1 h of professionally transcribed,
verbatim conversation (CORAAL), with **real ASR models**. Scoring is at the
**whole-recording** level: each model transcribes the entire interview and WER is
computed over the full transcript — so **no model is ever fed reference segment
boundaries** (no leakage), and Voxtral (which has no word timestamps) participates
naturally.

Branches (each a different architecture → complementary errors, §8):
* **A** Whisper large-v3-turbo (faster-whisper) — the baseline
* **B** wav2vec2 CTC (no LM; never hallucinates on silence)
* **Z** Qwen3-ASR-1.7B (the design's real backbone)
* **C** Voxtral Mini 3B (audio-LLM)
* **CW** CrisperWhisper 2.0 — a **verbatim**-tuned Whisper (keeps um/uh/false starts)
* **VV** Voxtral, **verbatim-prompted** (instruct mode, told to keep disfluencies)
* **phone** wav2vec2 phoneme CTC → **realized IPA** (the pronunciation path; its
  value is OOV/nonce recovery + PER, which CORAAL can't score — reported, not forced
  into word WER)

**Kaggle:** Settings → Accelerator → **GPU T4 x2**, Internet **ON**. Uses both T4s.

**Read the gap between the branch baseline and the oracle**, not absolute WER —
CORAAL is spontaneous English (high, deletion-heavy WER). Scoring folds
numbers/ordinals/spelling but keeps fillers (verbatim, §18/§29).
"""),
    md("""## 1. Install (from GitHub — nothing to upload)
`%pip` installs into the *running* kernel, so **Run All** completes in one click (with
`!pip`, Kaggle would stop after this cell because the env changed, needing a second
Run All). If it ever still stops here, just click Run All once more."""),
    code(f"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation OOM
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # Rust parallel-chunk downloads (2-5x faster)
# Persist ONLY the small CrisperWhisper venv across Kaggle sessions (Settings -> Persistence
# -> "Files only" keeps /kaggle/working). Model WEIGHTS are deliberately NOT persisted here:
# the persisted dir is capped (~20 GB) but the full model set is ~26 GB, so redirecting the HF
# cache into it fills the disk mid-run ("No space left on device"). Weights stay in the default
# cache on Kaggle's roomy ephemeral disk (re-downloaded each session, overlapped with compute by
# the §2a prefetch). Also delete any half-written weight cache an earlier version left in the
# persisted dir, so it stops eating the ~20 GB quota.
ASR_PERSIST = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("asr_persist")
os.environ["ASR_PERSIST"] = ASR_PERSIST
import shutil as _shutil
_shutil.rmtree(os.path.join(ASR_PERSIST, "hf_cache"), ignore_errors=True)   # reclaim space from the bad run
import torch
# TF32 matmul: free speedup on Ampere+ (Colab A100/L4), no-op on T4/Turing; inference-safe.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Everything up front (mid-notebook installs don't reliably import on Kaggle). Two calls:
# loose deps first, then PIN transformers/accelerate LAST so qwen-asr's required
# versions win over anything crisperwhisper/others pull in (4.57.6 also fits Whisper/Voxtral).
%pip -q install "mistral-common[audio]" phonemizer faster-whisper qwen-asr soundfile rapidfuzz hf_transfer virtualenv pyyaml typeguard
%pip -q install "transformers==4.57.6" "accelerate==1.12.0" "git+https://github.com/{GITHUB_REPO}.git"
# NOTE: CrisperWhisper is NOT installed here — its CT2 fork replaces the `ctranslate2`
# module and would clobber faster-whisper (branch A). It runs in an isolated venv (§9a).
import classroom_asr, os
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login; login(os.environ["HF_TOKEN"])
GPUS = list(range(torch.cuda.device_count())) or [None]
print("classroom_asr", classroom_asr.__version__, "| GPUs:", torch.cuda.device_count(),
      "|", [torch.cuda.get_device_name(g) for g in range(torch.cuda.device_count())])
"""),
    md("### Timing helper — every stage prints a persistent `⏱` line and is collected at the end"),
    code(r"""
import time as _time
TIMINGS = []                       # (stage_name, seconds); printed as a table in the last cell
def rec(name, dt):
    TIMINGS.append((name, dt)); print(f"⏱  {name}: {dt:.1f}s", flush=True); return dt
class stage:                       # `with stage("name"):` -> times the block, prints, records
    def __init__(self, name): self.name = name
    def __enter__(self): self.t0 = _time.time(); return self
    def __exit__(self, *a): rec(self.name, _time.time() - self.t0); return False
"""),
    md("## 2. Parameters"),
    code(r"""
COMPONENT, VERSION, AUDIO_PART = "les", "2021.07", "part01"   # or "prv","2018.10.06"
TARGET_MINUTES = 60

FW_MODEL       = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_VAD    = True      # skip silence (less hallucination); off = keep quiet words

USE_CTC        = True;  CTC_MODEL      = "facebook/wav2vec2-large-960h-lv60-self"
USE_QWEN3ASR   = True;  QWEN3ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
USE_VOXTRAL    = True;  VOXTRAL_MODEL  = "mistralai/Voxtral-Mini-3B-2507"
USE_PHONE      = True;  PHONE_MODEL    = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# PhoneticXeus (§7.3/§10): the design's named universal phone recognizer (XEUS + self-cond CTC,
# SOTA accented-English IPA). Phone branches output realized IPA (unscored on CORAAL — no
# phonetic reference); they feed the OOV/nonce recovery route (phone lattice -> P2G), not word WER.
USE_PHONETIC_XEUS = True;  PHONETIC_XEUS_MODEL = "changelinglab/PhoneticXeus"
# Verbatim branches (keep um/uh/false starts — the deletions clean models can't recover):
USE_CRISPER          = True;  CRISPER_SIZE = "large"   # turbo|large|medium|small (+ "_pro")
USE_VOXTRAL_VERBATIM = True             # Voxtral, prompted to transcribe verbatim

# Speed (all quality-neutral): the LLM branches auto-pick fp16 on T4 (Turing has no bf16
# tensor cores — that was most of the slowness; fp16 is the same inference path Whisper
# uses, no quality cost). Windows are batched. Voxtral's two passes share ONE 9.5 GB load.
# Defaults keep full quality: CrisperWhisper "large", BOTH Voxtral passes on.

# Window (seconds) for the audio-LLM branches (Qwen3, Voxtral). These emit a BOUNDED
# number of output tokens per call, so an over-long window truncates the transcript
# (600 s gave Qwen3 WER 0.87 — near-total deletion). Keep it near the utterance scale,
# like Whisper's internal 30 s window; this is capped by output budget, not GPU memory.
LLM_CHUNK_S    = 30

BASE = f"http://lingtools.uoregon.edu/coraal/{COMPONENT}/{VERSION}"
COMP = COMPONENT.upper()
"""),
    md("""## 2a. Prewarm in the background (overlap setup with compute)
Two serial blockers moved off the critical path: the isolated CrisperWhisper **venv build**
(§9a) and the **model downloads** (esp. Voxtral, ~9.5 GB). Both run in background threads
now, so they finish *while* the A→Qwen branches compute. Pure overlap — no GPU used here.

**Reused across runs:** with Kaggle **Settings → Persistence → "Files only"**, the small
CrisperWhisper **venv** (under `/kaggle/working`) survives between sessions, so it builds once
and later runs just reuse it. Model **weights are NOT persisted** — the full set (~26 GB)
exceeds the ~20 GB persisted-dir cap, so caching them there fills the disk mid-run; they live
in the roomy default cache and re-download each session (overlapped with compute by the
prefetch below). Transcription always runs fresh."""),
    code(r"""
import os, sys, subprocess, threading
# (a) CrisperWhisper CT2 venv — built in the background; §9a just joins this thread. Lives under
# the persisted dir (§1), so once built it's reused next session instead of rebuilt every run.
CW_WORK = os.path.join(os.environ.get("ASR_PERSIST", os.path.abspath(".")), "cw_iso")
os.makedirs(CW_WORK, exist_ok=True)
CW_VENV = os.path.join(CW_WORK, "venv"); CW_VENV_PY = os.path.join(CW_VENV, "bin", "python")
CW_READY = os.path.join(CW_WORK, ".venv_ready")   # sentinel: written only after pip install succeeds
_cw = {"thread": None, "err": None}
def _cw_prewarm():
    try:
        # Reuse only a FULLY-built venv. Checking the python binary alone is not enough: a run
        # that ran out of disk mid-install leaves the venv dir but no crisperwhisper -> the
        # sentinel guards against falsely "reusing" a broken build.
        if os.path.exists(CW_READY) and os.path.exists(CW_VENV_PY):
            print("CrisperWhisper venv: reusing persisted build", flush=True); return
        # virtualenv (not venv): seeds pip from bundled wheels, so it avoids the
        # ensurepip failure `python -m venv` hits on Kaggle. Reuses system torch.
        subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages", CW_VENV],
                       check=True)
        subprocess.run([os.path.join(CW_VENV, "bin", "pip"), "-q", "install",
                        "crisperwhisper[ct2]", "hf_transfer"], check=True)  # fast venv downloads
        open(CW_READY, "w").close()               # mark usable only now
    except Exception as e:
        _cw["err"] = e
if USE_CRISPER:
    _cw["thread"] = threading.Thread(target=_cw_prewarm, daemon=True); _cw["thread"].start()
    print("CrisperWhisper venv prewarm: started")

# (b) Prefetch model weights (hf_transfer-accelerated) so the heavy Voxtral/Qwen downloads
# overlap earlier branches instead of blocking their cells. Best-effort; branches re-fetch
# lazily if a prefetch fails.
def _prefetch():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return
    repos = [m for m, on in [
        (VOXTRAL_MODEL, USE_VOXTRAL or USE_VOXTRAL_VERBATIM), (QWEN3ASR_MODEL, USE_QWEN3ASR),
        (f"nyralabs/CrisperWhisper2.0_{CRISPER_SIZE}", USE_CRISPER),   # venv reads same HF cache
        (FW_MODEL, True), (CTC_MODEL, USE_CTC), (PHONE_MODEL, USE_PHONE)] if on]
    for r in repos:                                   # biggest first (Voxtral) for max overlap
        try: snapshot_download(r)
        except Exception as e: print("prefetch skip", r, repr(e)[:80])
threading.Thread(target=_prefetch, daemon=True).start()
print("model prefetch: started in background")
"""),
    md("## 3. Download + extract CORAAL"),
    code(r"""
import os, tarfile, urllib.request, ssl
from pathlib import Path
_dl0 = _time.time()
os.makedirs("coraal/txt", exist_ok=True); os.makedirs("coraal/audio", exist_ok=True)

def fetch(url, dest):
    if os.path.exists(dest): print("cached", dest); return
    print("downloading", url)
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f: f.write(r.read())
    print("  ->", os.path.getsize(dest)//1_000_000, "MB")

fetch(f"{BASE}/{COMP}_textfiles_{VERSION}.tar.gz", "coraal/txt.tar.gz")
fetch(f"{BASE}/{COMP}_audio_{AUDIO_PART}_{VERSION}.tar.gz", "coraal/audio.tar.gz")
for tarball, dest in [("coraal/txt.tar.gz","coraal/txt"), ("coraal/audio.tar.gz","coraal/audio")]:
    if any(Path(dest).iterdir()): continue
    with tarfile.open(tarball) as t: t.extractall(dest, filter="data")
print("transcripts:", len(list(Path('coraal/txt').rglob('*.txt'))),
      "| wavs:", len(list(Path('coraal/audio').rglob('*.wav'))))
rec("download+extract CORAAL", _time.time() - _dl0)
"""),
    md("## 4. Build interviews: full audio + full verbatim reference (~1h total)"),
    code(r"""
import torchaudio
from pathlib import Path
from classroom_asr.data.coraal import parse_transcript, iter_transcripts

wavs = {p.stem: p for p in Path("coraal/audio").rglob("*.wav")}
_full = {}
def load_16k(path):
    if path in _full: return _full[path]
    w, sr = torchaudio.load(str(path))
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
    a = w.squeeze(0).numpy().astype("float32"); _full[path] = a; return a

interviews = []   # (wav_path, full_reference_text)
total = 0.0
_bld0 = _time.time()
for txt in iter_transcripts("coraal/txt"):
    if txt.stem not in wavs: continue
    segs = sorted(parse_transcript(txt), key=lambda x: x.start)
    ref = " ".join(s.text for s in segs if s.text).strip()   # whole verbatim transcript
    if not ref: continue
    dur = len(load_16k(wavs[txt.stem])) / 16000
    interviews.append((wavs[txt.stem], ref)); total += dur
    if total >= TARGET_MINUTES * 60: break
refs = [r for _, r in interviews]
print(f"{len(interviews)} interviews, {total/60:.1f} min, "
      f"{sum(len(r.split()) for r in refs)} reference words")
rec("build interviews (load+resample)", _time.time() - _bld0)
"""),
    md("""## 5. Whole-recording runner + scoring (rapidfuzz)
`whole_rec` transcribes each interview on the GPUs. `wer_of` scores a branch over the
full transcripts; `oracle_wer` is the candidate-oracle at the word level — the fraction
of reference words that **no** branch recovers (a lower bound the pool can't beat)."""),
    code(r"""
import threading, gc, time
from tqdm.auto import tqdm
from rapidfuzz.distance import Levenshtein
from classroom_asr.normalize import Normalizer
SCORE = Normalizer(fold_numbers=True, fold_spelling=True)   # keep fillers; fold formatting

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Construct one model per GPU. This is SERIAL on purpose: transformers' from_pretrained
# mutates torch's *global* default dtype while materializing weights and restores it in a
# `finally`. Two loads in parallel threads race — one thread's restore fires mid-way through
# the other's materialization, leaving weights on the meta device ("Cannot copy out of meta
# tensor; no data!"). It's a race, so it only bit some runs. The heavy downloads are already
# prefetched in §2a, so a serial load is just the local disk->GPU copy (tens of seconds).
def load_models(make_model):
    return [make_model("cpu" if g is None else f"cuda:{g}") for g in GPUS]

def whole_rec(make_model, get_text, desc):
    out = [""] * len(interviews)
    t0 = time.time()
    models = load_models(make_model)               # parallel per-GPU load
    tload = time.time() - t0                       # includes first-time model download
    shards = [list(range(len(interviews)))[i::len(GPUS)] for i in range(len(GPUS))]
    def worker(model, sh, pos):
        for k in tqdm(sh, desc=f"{desc}:{pos}", position=pos, leave=True):  # leave=True: bar persists
            try: out[k] = get_text(model, load_16k(interviews[k][0]))
            except Exception as e: print(desc, "failed:", repr(e)[:120]); out[k] = ""
    t1 = time.time()
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(GPUS))]
    for t in ts: t.start()
    for t in ts: t.join()
    trun = time.time() - t1
    for m in models:
        if hasattr(m, "unload"):
            try: m.unload()
            except Exception as e: print(desc, "unload failed:", repr(e)[:100])
    models = None; del models; _free()   # release GPU memory before the next branch loads
    if torch.cuda.is_available():        # empty each device's cache, not just current
        for g in GPUS:
            if g is not None:
                with torch.cuda.device(g): torch.cuda.empty_cache()
    print(f"   {desc}: load {tload:.1f}s + transcribe {trun:.1f}s", flush=True)
    rec(desc, tload + trun)
    return out

def _free_models(models):
    for m in models:
        if hasattr(m, "unload"):
            try: m.unload()
            except Exception as e: print("unload failed:", repr(e)[:100])
    _free()
    if torch.cuda.is_available():
        for g in GPUS:
            if g is not None:
                with torch.cuda.device(g): torch.cuda.empty_cache()

def _windows_of(k, chunk_s):
    from classroom_asr.backends import iter_silence_chunks
    wav = load_16k(interviews[k][0])
    return [c for _, c in iter_silence_chunks(wav, 16000, chunk_s) if len(c) >= 400]

# Balance ALL ~chunk_s windows from ALL interviews evenly across the GPUs (so no GPU idles
# on the 2+1 interview split), transcribe, then reassemble each transcript in order. Output
# is identical to per-interview transcription (pure utilization win). `models` are preloaded
# (one per GPU) and NOT unloaded here (the caller owns them).
def window_pass(models, desc, *, chunk_s, batch_size):
    iv_chunks = [_windows_of(k, chunk_s) for k in range(len(interviews))]
    tasks = [(k, wi, c) for k, cs in enumerate(iv_chunks) for wi, c in enumerate(cs)]
    shards = [tasks[i::len(models)] for i in range(len(models))]
    results = {}
    def worker(model, shard, pos):
        cks = [t[2] for t in shard]
        if not cks: return
        try:
            texts = model.transcribe_chunk_list(cks, batch_size=batch_size)
        except Exception as e:
            print(desc, "failed:", repr(e)[:120]); texts = [""] * len(cks)
        for (k, wi, _), txt in zip(shard, texts): results[(k, wi)] = txt
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(models))]
    for t in ts: t.start()
    for t in ts: t.join()
    out = []
    for k, cs in enumerate(iv_chunks):
        parts = [results.get((k, wi), "") for wi in range(len(cs))]
        out.append(" ".join(p for p in parts if p).strip())
    return out

# window_pass with its own model load/unload + per-branch error guard + timing.
def run_windows(desc, make_model, *, chunk_s, batch_size):
    t0 = time.time()
    try:
        models = load_models(make_model)           # serial per-GPU load (thread-safe; see load_models)
    except Exception as e:
        print(f"[{desc:14s}] load FAILED: {repr(e)[:160]}"); _free(); return [""] * len(interviews)
    tload = time.time() - t0
    try:
        t1 = time.time()
        out = window_pass(models, desc, chunk_s=chunk_s, batch_size=batch_size)
        print(f"   {desc}: load {tload:.1f}s + transcribe {time.time()-t1:.1f}s", flush=True)
        return out
    except Exception as e:
        print(f"[{desc:14s}] FAILED: {repr(e)[:160]}"); return [""] * len(interviews)
    finally:
        _free_models(models)
        rec(desc, time.time() - t0)

_reftok = [SCORE.tokens(r) for r in refs]
def wer_of(hyps):
    E = R = 0
    for rt, h in zip(_reftok, hyps):
        E += Levenshtein.distance(rt, SCORE.tokens(h or "")); R += len(rt)
    return E / R if R else 0.0

def oracle_wer(pool):   # pool = list of branch hyp-lists; word-level recoverability
    unrec = R = 0
    for i, rt in enumerate(_reftok):
        hit = set()
        for hyps in pool:
            for tag, i0, i1, j0, j1 in Levenshtein.opcodes(rt, SCORE.tokens(hyps[i] or "")).as_list():
                if tag == "equal": hit.update(range(i0, i1))
        unrec += len(rt) - len(hit); R += len(rt)
    return unrec / R if R else 0.0

def add_branch(tag, hyps):
    # only count a branch if it actually produced text (a crashed branch is all "")
    got = sum(1 for h in hyps if h)
    if got == 0:
        print(f"[{tag:14s}] produced nothing (skipped from pool)"); return
    pool.append(hyps)
    print(f"[{tag:14s}] branch WER={wer_of(hyps):.3f}   oracle(pool)={oracle_wer(pool):.3f}"
          f"   ({got}/{len(hyps)} interviews)")

def run_branch(tag, make_model, get_text=None):
    get_text = get_text or (lambda m, a: m.transcribe_full(a))
    try:
        return whole_rec(make_model, get_text, tag)
    except Exception as e:
        # return empties (not None) so add_branch just skips this branch instead of crashing
        print(f"[{tag:14s}] FAILED to run: {repr(e)[:160]}"); _free()
        return [""] * len(interviews)
"""),
    md("## 6. Branch A — Whisper (whole recording) → baseline"),
    code(r"""
from classroom_asr.backends.faster_whisper_asr import FasterWhisperASR
# window-balanced: each interview split into large (~15 min) silence-snapped slices spread
# across both GPUs, so neither idles on the 2+1 interview split. Transcription is unchanged
# per slice (faster-whisper VAD+conditioning run inside each), so WER stays put.
hyp_A = run_windows("A whisper", lambda dev: FasterWhisperASR(
    FW_MODEL, language="en", device=dev, vad_filter=WHISPER_VAD),
    chunk_s=900, batch_size=1)
pool = []
add_branch("A", hyp_A)
print("^ baseline WER = branch A")
"""),
    md("## 7. Branch B — wav2vec2 CTC"),
    code(r"""
hyp_B = None
if USE_CTC:
    from classroom_asr.backends.wav2vec2_ctc import Wav2Vec2CTC
    hyp_B = run_windows("A+B", lambda dev: Wav2Vec2CTC(CTC_MODEL, device=dev),
                        chunk_s=900, batch_size=8)
    add_branch("A+B", hyp_B)
"""),
    md("## 8. Branch Z — Qwen3-ASR-1.7B (the design's backbone)"),
    code(r"""
hyp_Z = None
if USE_QWEN3ASR:
    from classroom_asr.backends.qwen3_asr import Qwen3ASR
    # windows balanced across GPUs (no 2+1 idle), reassembled in order (same output)
    hyp_Z = run_windows("A+B+Qwen3",
                        lambda dev: Qwen3ASR(QWEN3ASR_MODEL, language="English", device=dev),
                        chunk_s=LLM_CHUNK_S, batch_size=32)   # 64 gave no speedup, 2x VRAM -> back to 32
    add_branch("A+B+Qwen3", hyp_Z)
"""),
    md("""## 9. Branch C / VV — Voxtral Mini 3B, **both passes on one load**
Voxtral runs twice — clean transcription and verbatim-prompted (instruct mode, told to
keep every filler/false start/repetition) — but the 9.5 GB weights are loaded **once**
and reused for both passes (quality-neutral: same model, only the decode mode changes).
Both are scored so we see the clean vs verbatim oracle contribution side by side."""),
    code(r"""
hyp_C = hyp_VV = None
if USE_VOXTRAL or USE_VOXTRAL_VERBATIM:
    from classroom_asr.backends.voxtral_asr import VoxtralASR
    with stage("Voxtral load (shared)"):            # one 9.5 GB load reused by both passes
        vmodels = load_models(lambda dev: VoxtralASR(VOXTRAL_MODEL, language="en", device=dev))
    def _vox_pass(mode, tag):                       # windows balanced across GPUs, one mode
        for m in vmodels: m.mode = mode
        t0 = time.time(); out = window_pass(vmodels, tag, chunk_s=LLM_CHUNK_S, batch_size=12)
        rec(tag, time.time() - t0); return out
    if USE_VOXTRAL:
        hyp_C = _vox_pass("transcription", "+Voxtral"); add_branch("+Voxtral", hyp_C)
    if USE_VOXTRAL_VERBATIM:
        hyp_VV = _vox_pass("verbatim", "+VoxtralVerbatim"); add_branch("+VoxtralVerbatim", hyp_VV)
    _free_models(vmodels); vmodels = None; del vmodels
"""),
    md("""## 9a. Branch CW — CrisperWhisper 2.0 (**verbatim** Whisper, isolated CT2)
The verbatim lever: a Whisper fine-tune that *keeps* the `um`/`uh`/false starts the clean
branches delete. Its fast **CT2 runtime uses a forked `ctranslate2`** that can't share
site-packages with faster-whisper (branch A) — so CrisperWhisper runs in a
`--system-site-packages` **venv** (reuses the main torch/transformers, shadows only
`ctranslate2` with the fork) driven by a subprocess. The venv was built in the background
back in §2a, so this cell usually just joins it and transcribes. Branch A keeps stock CT2;
the main kernel never imports the fork. **Both GPUs are used**: one pinned subprocess per GPU
transcribes a disjoint slice of the interviews in parallel (whole files, so each keeps
CrisperWhisper's native chunking and the output is identical). The worker prints which backend
it got — `ct2 (fast)` or the `transformers (SLOW)` fallback — so a slow run is visible."""),
    code(r"""
hyp_CW = None
if USE_CRISPER:
    import os, sys, json, subprocess, threading
    _cw_t0 = time.time()
    WORKER = os.path.join(CW_WORK, "cw_worker.py")
    try:
        if _cw["thread"] is not None:
            _cw["thread"].join()                 # wait for the background venv build (§2a)
        if _cw["err"] is not None:
            raise _cw["err"]
        with open(WORKER, "w") as f:
            f.write('''
import sys, json, warnings, logging
from crisperwhisper import CrisperWhisperModel   # imports transformers -> its loggers now exist
# The crisperwhisper package still calls from_pretrained with the old `torch_dtype=` kwarg;
# that is inside the dependency, so it can't be fixed at our source. transformers emits it via
# its LOGGER (warning_once), not warnings.warn. Attach the filter to the EMITTING logger
# (transformers.modeling_utils): Logger.handle runs a logger's own filters before it dispatches
# to any handler, so this drops just that one line regardless of whether/when a handler exists
# (the earlier handler-based filter missed it -- in a subprocess the record hits logging's
# lastResort handler, which lives on no logger). Not a verbosity change, not a broad mute.
warnings.filterwarnings("ignore", message=".*torch_dtype.*")   # belt-and-suspenders
class _DropTorchDtype(logging.Filter):
    def filter(self, record): return "torch_dtype" not in record.getMessage()
_flt = _DropTorchDtype()
for _name in ("transformers", "transformers.modeling_utils"):
    logging.getLogger(_name).addFilter(_flt)
size, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
paths = json.load(open(inp))
try:
    m = CrisperWhisperModel(size, backend="ct2")          # fast forked CT2
    print("CW backend: ct2 (fast)", flush=True)           # to STDOUT so we can see the path taken
except Exception as e:
    print("CW backend: transformers (SLOW) -- ct2 failed:", repr(e)[:200], flush=True)
    m = CrisperWhisperModel(size, backend="transformers")  # in-venv fallback
out = {}
for p in paths:
    try:
        r = m.transcribe(p, language="en"); out[p] = getattr(r, "text", "") or ""
    except Exception as e:
        out[p] = ""; print("fail", p, repr(e)[:120], file=sys.stderr)
json.dump(out, open(outp, "w"))
''')
        paths = [os.path.abspath(str(interviews[k][0])) for k in range(len(interviews))]
        # One worker per GPU, pinned via CUDA_VISIBLE_DEVICES, run in parallel so both GPUs
        # transcribe instead of one sitting idle. Whole files are split across GPUs (no
        # cross-file interaction and each file keeps CrisperWhisper's native internal 30s
        # chunking), so the per-file output is byte-identical to the single-worker run.
        gpus = [g for g in GPUS if g is not None] or [None]
        shards = [paths[i::len(gpus)] for i in range(len(gpus))]
        res = {}
        def _cw_run(gi, gpu, shard):
            if not shard:
                return
            inp = os.path.join(CW_WORK, f"in_{gi}.json"); outp = os.path.join(CW_WORK, f"out_{gi}.json")
            json.dump(shard, open(inp, "w"))
            env = dict(os.environ)
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)      # each subprocess sees exactly one GPU
            subprocess.run([CW_VENV_PY, WORKER, CRISPER_SIZE, inp, outp], check=True, env=env)
            res.update(json.load(open(outp)))               # disjoint keys per shard -> safe
        _cw_ts = [threading.Thread(target=_cw_run, args=(gi, gpus[gi], shards[gi]))
                  for gi in range(len(gpus))]
        for t in _cw_ts: t.start()
        for t in _cw_ts: t.join()
        from classroom_asr.backends.crisperwhisper_asr import CrisperWhisperV2
        hyp_CW = [CrisperWhisperV2._clean(res.get(p, "")) for p in paths]   # [um]->um, drop [laughter]
    except Exception as e:
        print("CrisperWhisper isolation failed:", repr(e)[:200]); hyp_CW = [""] * len(interviews)
    rec("+CrisperWhisper", time.time() - _cw_t0)
    add_branch("+CrisperWhisper", hyp_CW)
"""),
    md("""## 10. Phone branches — realized IPA (the pronunciation path)
Not a word transcript: the phone branch's product is *pronunciation*. Scoring it needs a
phonetic reference (PER/IPA-CER, §18.1), which CORAAL doesn't provide — so we show the
realized IPA rather than force it through a naive P2G into a misleading word WER. It's a
first-class part of the design (OOV/nonce recovery), just not exercised by this dataset.

Two models: `wav2vec2-lv-60-espeak` (now decoded via a manual vocab CTC decoder — the
`Wav2Vec2PhonemeCTCTokenizer` won't load on transformers 4.57.x, so the branch used to be
skipped), and **PhoneticXeus** — the design's named universal recognizer (XEUS + self-conditioned
CTC, SOTA accented-English IPA). Compare their IPA on the same audio."""),
    code(r"""
if USE_PHONE:
    try:                                    # illustrative + unscored: never let it stop the run
        from classroom_asr.backends.wav2vec2_phone import Wav2Vec2Phone
        ipa = whole_rec(lambda dev: Wav2Vec2Phone(PHONE_MODEL, device=dev),
                        lambda m, a: m.transcribe_full(a), "phone")
        print("wav2vec2-espeak realized IPA (first 300 chars of interview 0):")
        print(" ", (ipa[0] or "")[:300])
        print("\nreference words (for contrast):", refs[0][:200])
    except Exception as e:
        print("phone branch skipped:", repr(e)[:160])

if USE_PHONETIC_XEUS:
    try:                                    # the design's named phone model (SOTA IPA); unscored
        from classroom_asr.backends.phonetic_xeus import PhoneticXeus
        xipa = whole_rec(lambda dev: PhoneticXeus(PHONETIC_XEUS_MODEL, device=dev),
                         lambda m, a: m.transcribe_full(a), "PhoneticXeus")
        print("PhoneticXeus realized IPA (first 300 chars of interview 0):")
        print(" ", (xipa[0] or "")[:300])
    except Exception as e:
        print("PhoneticXeus skipped:", repr(e)[:200])
"""),
    md("## 11. What specific mistakes — error analysis (branch A vs reference)"),
    code(r"""
from collections import Counter
# rapidfuzz opcodes so whole-interview alignment stays fast
dels, subs, ins = Counter(), Counter(), Counter()
S = D = I = R = 0
for r, h in zip(refs, hyp_A or []):
    rt, ht = SCORE.tokens(r), SCORE.tokens(h or ""); R += len(rt)
    for tag, i0, i1, j0, j1 in Levenshtein.opcodes(rt, ht).as_list():
        if tag == "replace":
            for a, b in zip(rt[i0:i1], ht[j0:j1]): subs[f"{a} -> {b}"] += 1; S += 1
        elif tag == "delete":
            for a in rt[i0:i1]: dels[a] += 1; D += 1
        elif tag == "insert":
            for b in ht[j0:j1]: ins[b] += 1; I += 1
E = S + D + I
if not R or not E:
    print("branch A produced nothing — nothing to analyse"); raise SystemExit
print(f"branch-A WER {E/R:.3f}  |  S={S} ({100*S//E}%)  D={D} ({100*D//E}%)  I={I} ({100*I//E}%)")
print("\nMost-deleted reference words (what Whisper drops):")
for w, n in dels.most_common(15): print(f"   {w!r:18} x{n}")
print("\nMost-common substitutions (ref -> hyp):")
for p, n in subs.most_common(15): print(f"   {p:30} x{n}")
print("\nMost-inserted words (hyp words with no ref):")
for w, n in ins.most_common(15): print(f"   {w!r:18} x{n}")
"""),
    md("""## 11.5 Oracle-floor breakdown — what the *whole ensemble* still misses
Section 11 is branch A alone. This is the **oracle floor**: the reference words that **no
branch** recovered — the exact `oracle_wer` set (a ref word counts as recovered iff it lands
in an `equal` span for at least one branch). It's the honest ceiling for *this* model set.

Vernacular is deliberately **kept as errors** (CORAAL is regional AAL — `gonna`, `y'all`,
g-dropping are the features of interest, not noise to fold away), so they surface here as
genuine misses. The category split separates convention/vernacular from the model-limited
deletions (backchannel overlap, reduced function words)."""),
    code(r"""
from collections import Counter
# Same "recovered" rule as oracle_wer(): a ref index is recovered iff some branch aligns it
# in an `equal` span. Everything else is the floor — tally those ref words by type/category.
floor = Counter(); total_unrec = R = 0
if not pool:
    print("no branches in pool — nothing to break down")
else:
    for i, rt in enumerate(_reftok):
        hit = set()
        for hyps in pool:
            for tag, i0, i1, j0, j1 in Levenshtein.opcodes(rt, SCORE.tokens(hyps[i] or "")).as_list():
                if tag == "equal": hit.update(range(i0, i1))
        for k, w in enumerate(rt):
            if k not in hit: floor[w] += 1; total_unrec += 1
        R += len(rt)

    # buckets (mutually exclusive, first match wins). post-normalization forms: edge
    # apostrophes are already stripped by SCORE.tokens (so 'cause -> cause), internal kept.
    FILLER = {"um","uh","uhm","erm","er","mm","hmm","hm","mmhm","mhm","mm-hmm","uh-huh","huh",
              "ah","oh","eh","yeah","yep","yup","nah","mkay","naw"}
    VERNAC = {"gonna","wanna","gotta","finna","tryna","imma","gon","y'all","yall","ain't","aint",
              "o'clock","cause","cuz","lemme","gimme","kinda","sorta","dunno","bout","em",
              "goin","doin","nothin","somethin","talkin","gettin","comin","tryin"}
    FUNC   = {"i","a","an","the","and","to","of","in","is","was","that","you","it","he","she","we",
              "they","on","so","at","my","me","be","do","as","but","or","if","for","with","this",
              "your","our","his","her","them","then","there","have","had","has","are","were","just"}
    def cat(w):
        if w in VERNAC: return "vernacular (kept as errors by design)"
        if w in FILLER: return "filler/backchannel"
        if w in FUNC or len(w) <= 2: return "short/function word"
        return "content word"

    bycat = Counter()
    for w, n in floor.items(): bycat[cat(w)] += n
    print(f"ORACLE FLOOR: {total_unrec} of {R} ref words recovered by NO branch"
          f"   (oracle WER {total_unrec/R:.3f}; matches the pool oracle above)")
    print("\nby category (share of the floor):")
    for c, n in bycat.most_common():
        print(f"   {n:5d}  {100*n/total_unrec:4.1f}%   {c}")
    print("\nMost-common unrecovered reference words (what the whole ensemble misses):")
    for w, n in floor.most_common(30):
        print(f"   {w!r:16} x{n:<4} [{cat(w)}]")
"""),
    md("""## 11.6 Selection — deterministic ROVER fusion (the LLM selector's substrate)
The oracle (0.073) is the *ceiling* if you always pick the right branch per word; the baseline
(branch A, ~0.193) is one branch alone. A real system has no reference, so it aligns the
branches to each other and votes. This is that layer: `rover.fuse` builds the per-word candidate
graph (each slot = every branch's aligned candidate, `NULL` = drop) and majority-votes it — a
strong no-LLM baseline that also settles the confident, agreeing spans (§12.5). The design's
Qwen3.5-9B judge (§14) is a drop-in `chooser` that overrides only the *uncertain* slots; this
number is what it has to beat, on the way from baseline down toward the oracle floor."""),
    code(r"""
from classroom_asr.rover import fuse, build_graph
# every whole-recording WORD branch we actually produced (phone branches are IPA, excluded)
_wb = [globals().get(n) for n in ["hyp_A", "hyp_B", "hyp_Z", "hyp_C", "hyp_VV", "hyp_CW"]]
_wb = [h for h in _wb if h and any(h)]
fused = [fuse([h[k] for h in _wb], norm=SCORE) for k in range(len(interviews))]
fused_wer = wer_of(fused)
# how many slots the LLM would even be asked about (disagreement), vs frozen-confident ones
_disagree = _total = 0
for k in range(len(interviews)):
    for s in build_graph([SCORE.tokens(h[k]) for h in _wb if h[k]]):
        _total += 1; _disagree += 0 if s.agreed else 1
print(f"fused {len(_wb)} word branches")
print(f"[FUSED (ROVER)  ] final WER={fused_wer:.3f}"
      f"   vs baseline A={wer_of(hyp_A):.3f}  vs oracle floor={oracle_wer(pool):.3f}")
print(f"headroom captured by deterministic vote: "
      f"{100*(wer_of(hyp_A)-fused_wer)/max(wer_of(hyp_A)-oracle_wer(pool),1e-9):.0f}% of baseline->oracle gap")
print(f"uncertain slots (what the LLM judge would see): {_disagree}/{_total} "
      f"({100*_disagree/max(_total,1):.0f}%); the rest are frozen-confident (§12.5)")
"""),
    md("""## 12. Timings + save summary
Persistent per-stage wall-times (every stage `⏱`-printed as it finished; here they're
collected into one table) and the WER/oracle summary."""),
    code(r"""
import json
# --- timing table (slowest first) ---
print("=== wall-time by stage ===")
for name, dt in sorted(TIMINGS, key=lambda x: -x[1]):
    print(f"   {dt:7.1f}s  {name}")
print(f"   {sum(dt for _, dt in TIMINGS):7.1f}s  TOTAL (sum of tracked stages)")

res = {}
for name, h in [("A_whisper", hyp_A), ("B_ctc", hyp_B), ("Z_qwen3", hyp_Z), ("C_voxtral", hyp_C)]:
    if h and any(h): res[name] = round(wer_of(h), 4)
summary = {"component": COMPONENT, "interviews": len(interviews), "minutes": round(total/60, 1),
           "scoring": "whole-recording; numbers+spelling folded; fillers kept",
           "branch_wer": res, "oracle_wer": round(oracle_wer(pool), 4),
           "baseline_wer": res.get("A_whisper"),
           "fused_rover_wer": round(fused_wer, 4) if "fused_wer" in dir() else None,
           "timings_s": {name: round(dt, 1) for name, dt in TIMINGS}}
try:   # oracle-floor breakdown from §11.5 (defined only if the pool had branches)
    summary["oracle_floor_by_category"] = dict(bycat.most_common())
    summary["oracle_floor_top"] = [[w, n] for w, n in floor.most_common(30)]
except NameError:
    pass
json.dump(summary, open("coraal_oracle_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
"""),
]

NB = {"cells": CELLS,
      "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "| installs from github.com/" + GITHUB_REPO)
