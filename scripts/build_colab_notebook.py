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
OUT = ROOT / "colab" / "CORAAL_candidate_oracle_payload.ipynb"


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
* **A3** Whisper large-v3 (faster-whisper, FP16 beam 5) — quality shadow
* **B** wav2vec2 CTC (no LM; never hallucinates on silence)
* **Z** Qwen3-ASR-1.7B (the design's real backbone)
* **C** Voxtral Mini 3B (audio-LLM)
* **CW** CrisperWhisper 2.0 — a **verbatim**-tuned Whisper (keeps um/uh/false starts)
* **CWV** Crisper Verbatimize — Qwen content conditioned on each matching audio window
* **VV** Voxtral, **verbatim-prompted** (instruct mode, told to keep disfluencies)
* **phone** wav2vec2 phoneme CTC + PhoneticXEUS → timestamped **realized-IPA evidence** when
  the selector is enabled (currently paused during word-branch overlap analysis)

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
import os, time as _run_time
RUN_STARTED_EPOCH = _run_time.time()          # includes installs, downloads, and every branch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation OOM
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"     # current high-throughput Hub/Xet downloads
ASR_GIT_REF = os.environ.get("CLASSROOM_ASR_GIT_REF", "main")
# Persist the CrisperWhisper venv and its bounded converted CT2 main model across Kaggle
# sessions (Settings -> Persistence -> "Files only" keeps /kaggle/working). The full ensemble's
# source model WEIGHTS are deliberately NOT persisted here:
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
%pip -q install "mistral-common[audio]" phonemizer faster-whisper qwen-asr soundfile rapidfuzz virtualenv pyyaml typeguard
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.57.6", "accelerate==1.12.0",
                f"git+https://github.com/{GITHUB_REPO}.git@{{ASR_GIT_REF}}"], check=True)
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
FW_QUALITY_MODEL = "Systran/faster-whisper-large-v3"
USE_WHISPER_LARGE_V3 = True
WHISPER_VAD    = True      # skip silence (less hallucination); off = keep quiet words

USE_CTC        = True;  CTC_MODEL      = "facebook/wav2vec2-large-960h-lv60-self"
USE_QWEN3ASR   = True;  QWEN3ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
USE_VOXTRAL    = True;  VOXTRAL_MODEL  = "mistralai/Voxtral-Mini-3B-2507"
# Selector paused while the word branches undergo leave-one-out overlap analysis. Phone/IPA is
# selector input in this benchmark, so skip both phone models too rather than compute unused data.
USE_LLM_SELECTOR = False; SELECTOR_MODEL = "Qwen/Qwen3.5-9B"
USE_PHONE      = USE_LLM_SELECTOR; PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# PhoneticXeus (§7.3/§10): the design's named universal phone recognizer (XEUS + self-cond CTC,
# SOTA accented-English IPA). Phone branches output timestamped realized-IPA evidence (unscored
# on CORAAL — no phonetic reference) that is retrieved into the selector prompt, not merely printed.
USE_PHONETIC_XEUS = USE_LLM_SELECTOR
PHONETIC_XEUS_MODEL = "changelinglab/PhoneticXeus"
# Verbatim branches (keep um/uh/false starts — the deletions clean models can't recover):
USE_CRISPER          = True;  CRISPER_SIZE = "large"   # turbo|large|medium|small (+ "_pro")
CRISPER_VERSION      = "2.0.2"  # exact CT2 runtime validated by the notebook integration
USE_CRISPER_VERBATIZE = USE_CRISPER and USE_QWEN3ASR  # Qwen content + acoustic disfluencies
USE_VOXTRAL_VERBATIM = True             # Voxtral, prompted to transcribe verbatim
# Conversation-LLM selector (§14): the design's judge over the candidate graph. Overrides only
# the contested slots (confident spans frozen, §12.5), driving the final transcript from the
# branch baseline toward the oracle floor. Qwen3.5-9B = design's "default first judge" (§14.2);
# runs isolated (own venv, newer transformers) in fp16 across both freed T4s. NEW hatch off (§14.5).
# Pin the selector-only environment independently from qwen-asr's older main-kernel stack.
# These are the versions validated by this notebook; the fingerprint in §2a upgrades a persisted
# selector venv whenever either pin changes instead of silently reusing stale packages.
SELECTOR_TRANSFORMERS = "5.15.0"; SELECTOR_ACCELERATE = "1.14.0"

# Speed (all quality-neutral): the LLM branches auto-pick fp16 on T4 (Turing has no bf16
# tensor cores — that was most of the slowness; fp16 is the same inference path Whisper
# uses, no quality cost). Windows are batched. Voxtral's two passes share ONE 9.5 GB load.
# Defaults keep full quality: CrisperWhisper "large", BOTH Voxtral passes on.

# The 90 s Qwen shadow regressed WER 0.157 -> 0.181 and ran slower. Restore the known-good
# silence-snapped 30 s windows and isolate automatic language detection as the only Qwen decode
# change in the next run. Larger-context continuity needs a boundary-aware method, not merely a
# larger independent request. 512 tokens is ample for a 30 s verbatim window.
QWEN_CHUNK_S = 30
QWEN_MAX_NEW_TOKENS = 512
VOXTRAL_CHUNK_S = 30

BASE = f"http://lingtools.uoregon.edu/coraal/{COMPONENT}/{VERSION}"
COMP = COMPONENT.upper()
"""),
    md("""## 2a. Prewarm in the background (overlap setup with compute)
Two serial blockers moved off the critical path: the isolated CrisperWhisper **venv build**
(§9a) and the **model downloads** (esp. Voxtral, ~9.5 GB). Both run in background threads
now, so they finish *while* the A→Qwen branches compute. Pure overlap — no GPU used here.

**Reused across runs:** with Kaggle **Settings → Persistence → "Files only"**, the small
CrisperWhisper **venv** and its converted CT2 main model (under `/kaggle/working/cw_iso`)
survive between sessions. Source model **weights are NOT persisted** — the full ensemble exceeds
the persisted-dir cap, so those live in the roomy default cache and re-download each session
(overlapped with compute by the prefetch below). Once CrisperWhisper's persisted CT2 conversion is
complete, its now-redundant 1.62 GB source checkpoint is no longer fetched. No transcript,
timestamp, IPA, phone, selector, or other derived inference output is cached; transcription always
runs fresh."""),
    code(r"""
import os, sys, subprocess, threading, shutil, glob, hashlib
# A persisted venv can come back from a previous Kaggle session with bin/python stripped of its
# execute bit (persistence doesn't preserve +x) -> PermissionError at run time. So before reusing
# one, prove its python actually runs; if not, chmod, then wipe-and-rebuild as a last resort.
def _venv_ok(py):
    try:
        return subprocess.run([py, "-c", ""], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False
def _reusable(ready, py, venvdir):
    if not (os.path.exists(ready) and os.path.exists(py)):
        return False
    if _venv_ok(py):
        return True
    subprocess.run(["chmod", "-R", "u+rwx", os.path.join(venvdir, "bin")], capture_output=True)
    if _venv_ok(py):
        return True
    shutil.rmtree(venvdir, ignore_errors=True)          # broken beyond a chmod -> force a rebuild
    try: os.remove(ready)
    except OSError: pass
    return False
# (a) CrisperWhisper CT2 venv — built in the background; §9a just joins this thread. Lives under
# the persisted dir (§1), so once built it's reused next session instead of rebuilt every run.
CW_WORK = os.path.join(os.environ.get("ASR_PERSIST", os.path.abspath(".")), "cw_iso")
os.makedirs(CW_WORK, exist_ok=True)
CW_VENV = os.path.join(CW_WORK, "venv"); CW_VENV_PY = os.path.join(CW_VENV, "bin", "python")
CW_READY = os.path.join(CW_WORK, ".venv_ready")   # stores the exact validated environment spec
CW_ENV_SPEC = f"crisperwhisper[ct2]=={CRISPER_VERSION}"
CW_CT2_CACHE = os.path.join(CW_WORK, "ct2_models")
os.makedirs(CW_CT2_CACHE, exist_ok=True)
def _cw_converted_ready():
    # True only for a completed persisted CT2 conversion, never a partial directory.
    repo = (CRISPER_SIZE if "/" in CRISPER_SIZE
            else f"nyralabs/CrisperWhisper2.0_{CRISPER_SIZE}")
    slug = repo.replace("/", "--").replace("\\", "--")
    key = f"{slug}_float16_{hashlib.sha256(repo.encode()).hexdigest()[:12]}"
    candidate = os.path.join(CW_CT2_CACHE, key)
    return (os.path.isfile(os.path.join(candidate, "model.bin"))
            and os.path.isfile(os.path.join(candidate, ".conversion_complete")))
# Remove derived handoff files left by payloads before they were moved to session-temporary storage.
# These exact non-recursive targets never include the reusable venv or converted model cache.
for _legacy in ([os.path.join(CW_WORK, "cw_worker.py"),
                 os.path.join(CW_WORK, "ct2_model_init.lock")]
                + glob.glob(os.path.join(CW_WORK, "in_*.json"))
                + glob.glob(os.path.join(CW_WORK, "out_*.json"))):
    try: os.remove(_legacy)
    except OSError: pass
_cw = {"thread": None, "err": None}
def _cw_runtime_ok():
    probe = (
        "import importlib.metadata; from crisperwhisper import CrisperWhisperModel; "
        f"assert importlib.metadata.version('crisperwhisper') == '{CRISPER_VERSION}'; "
        "assert 'cache_dir' in __import__('inspect').signature(CrisperWhisperModel).parameters"
    )
    try:
        return subprocess.run([CW_VENV_PY, "-c", probe], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False
def _cw_prewarm():
    try:
        # Reuse only a fully-built, actually-runnable venv (sentinel guards a mid-install crash;
        # _reusable guards a persisted venv whose python lost +x).
        try:
            _ready_spec = open(CW_READY).read().strip()
        except OSError:
            _ready_spec = ""
        if (_reusable(CW_READY, CW_VENV_PY, CW_VENV)
                and _ready_spec == CW_ENV_SPEC and _cw_runtime_ok()):
            print("CrisperWhisper venv: reusing persisted build", flush=True); return
        # virtualenv (not venv): seeds pip from bundled wheels, so it avoids the
        # ensurepip failure `python -m venv` hits on Kaggle. Reuses system torch.
        subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages", CW_VENV],
                       check=True)
        subprocess.run([os.path.join(CW_VENV, "bin", "pip"), "-q", "install", "-U",
                        CW_ENV_SPEC], check=True)
        if not _cw_runtime_ok():
            raise RuntimeError(f"CrisperWhisper runtime smoke check failed: {CW_ENV_SPEC}")
        with open(CW_READY, "w") as f: f.write(CW_ENV_SPEC)  # mark usable only now
    except Exception as e:
        _cw["err"] = e
if USE_CRISPER:
    _cw["thread"] = threading.Thread(target=_cw_prewarm, daemon=True); _cw["thread"].start()
    print("CrisperWhisper venv prewarm: started")

# (a2) Selector venv — Qwen3.5-9B needs a newer transformers than the 4.57.6 the main kernel pins
# for qwen-asr, so the LLM judge runs in its OWN --system-site-packages venv (shadows
# transformers/accelerate; reuses system torch + the installed classroom_asr).
# Built in the background so it's ready when the acoustic branches finish (§11.7).
SEL_WORK = os.path.join(os.environ.get("ASR_PERSIST", os.path.abspath(".")), "sel_iso")
os.makedirs(SEL_WORK, exist_ok=True)
SEL_VENV = os.path.join(SEL_WORK, "venv"); SEL_VENV_PY = os.path.join(SEL_VENV, "bin", "python")
SEL_READY = os.path.join(SEL_WORK, ".venv_ready")
for _legacy_name in ("sel_worker.py", "in.json", "out.json"):
    try: os.remove(os.path.join(SEL_WORK, _legacy_name))
    except OSError: pass
SEL_ENV_SPEC = f"transformers=={SELECTOR_TRANSFORMERS}|accelerate=={SELECTOR_ACCELERATE}"
_sel = {"thread": None, "err": None}
def _sel_runtime_ok():
    # A matching sentinel is insufficient if persistence restored a partial/corrupt environment.
    # Import the exact load-bearing API before accepting the venv or marking it ready. Cold imports
    # of torch + Transformers can exceed a minute while model prefetch is saturating Kaggle I/O, so
    # do not turn a slow-but-valid import into a permanent selector failure.
    probe = (
        "import accelerate, torch, transformers; "
        "from transformers import AutoModelForMultimodalLM, AutoProcessor; "
        "from transformers.utils import is_torch_available; assert is_torch_available(); "
        f"assert transformers.__version__ == '{SELECTOR_TRANSFORMERS}'; "
        f"assert accelerate.__version__ == '{SELECTOR_ACCELERATE}'"
    )
    try:
        result = subprocess.run([SEL_VENV_PY, "-c", probe], capture_output=True,
                                text=True, timeout=300)
        if result.returncode == 0:
            return None
        return (result.stderr or result.stdout or f"exit {result.returncode}").strip()[-2400:]
    except Exception as exc:
        return repr(exc)
def _sel_prewarm():
    try:
        try:
            _ready_spec = open(SEL_READY).read().strip()
        except OSError:
            _ready_spec = ""
        if (_reusable(SEL_READY, SEL_VENV_PY, SEL_VENV)
                and _ready_spec == SEL_ENV_SPEC and _sel_runtime_ok() is None):
            print("selector venv: reusing persisted build", flush=True); return
        if not _venv_ok(SEL_VENV_PY):
            subprocess.run([sys.executable, "-m", "virtualenv", "--system-site-packages", SEL_VENV],
                           check=True)
        subprocess.run([os.path.join(SEL_VENV, "bin", "pip"), "-q", "install", "-U",
                        f"transformers=={SELECTOR_TRANSFORMERS}",
                        f"accelerate=={SELECTOR_ACCELERATE}"], check=True)   # fp16, no bnb
        _runtime_problem = _sel_runtime_ok()
        if _runtime_problem is not None:
            raise RuntimeError(f"selector runtime smoke check failed: {SEL_ENV_SPEC}: "
                               f"{_runtime_problem}")
        with open(SEL_READY, "w") as f: f.write(SEL_ENV_SPEC)
    except Exception as e:
        _sel["err"] = e
        print("selector venv prewarm failed:", repr(e)[:1600], flush=True)
if USE_LLM_SELECTOR:
    _sel["thread"] = threading.Thread(target=_sel_prewarm, daemon=True); _sel["thread"].start()
    print("selector venv prewarm: started")

# (b) Prefetch model weights (high-performance Xet) so downloads overlap compute instead of
# blocking branch cells. This caches model files only, never transcripts or derived evidence.
# Best-effort; branches re-fetch lazily if a prefetch fails.
def _prefetch():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return
    # Fetch in first-use order so A/B/Z do not wait behind the largest late-stage models. The
    # selector remains far enough ahead of §11.7 because all acoustic inference runs before it.
    repos = [m for m, on in [
        (FW_MODEL, True), (FW_QUALITY_MODEL, USE_WHISPER_LARGE_V3),
        (CTC_MODEL, USE_CTC), (QWEN3ASR_MODEL, USE_QWEN3ASR),
        (VOXTRAL_MODEL, USE_VOXTRAL or USE_VOXTRAL_VERBATIM),
        # The persisted CT2 directory is self-contained. Fetch the 1.62 GB source only when a
        # conversion is actually missing; workers otherwise load the local model directly.
        (f"nyralabs/CrisperWhisper2.0_{CRISPER_SIZE}",
         USE_CRISPER and not _cw_converted_ready()),
        (PHONE_MODEL, USE_PHONE), (PHONETIC_XEUS_MODEL, USE_PHONETIC_XEUS),
        (SELECTOR_MODEL, USE_LLM_SELECTOR)] if on]
    for r in repos:
        try:
            # The runtime is PyTorch-only. Several wav2vec2 repositories also contain complete
            # TensorFlow and Flax copies (2.52 GB combined for branch B); never download them.
            snapshot_download(r, ignore_patterns=["*.h5", "*.msgpack"])
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
`whole_rec` transcribes each interview on the GPUs. `wer_of` scores a branch over the full
transcripts; `recall_floor` = fraction of reference words **no** branch produced (a recall lower
bound — ignores insertions); `realizable_oracle_wer` = the best full WER a selector over the
candidate graph could actually reach (§11.6)."""),
    code(r"""
import threading, gc, time
from tqdm.auto import tqdm
from rapidfuzz.distance import Levenshtein
from classroom_asr.normalize import Normalizer
SCORE = Normalizer(fold_numbers=True, fold_spelling=True)   # keep fillers; fold formatting
_GRAPH_OPCODE_CACHE = {}

def fast_graph_opcodes(pivot, hypothesis):
    # Cached compiled alignment for repeated whole-interview candidate-graph builds.
    key = (tuple(pivot), tuple(hypothesis))
    if key not in _GRAPH_OPCODE_CACHE:
        _GRAPH_OPCODE_CACHE[key] = Levenshtein.opcodes(pivot, hypothesis).as_list()
    return _GRAPH_OPCODE_CACHE[key]

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

WINDOW_PARTS = {}          # desc -> per-interview timestamped text windows; selector anchor
_WINDOW_RECORD_CACHE = {}  # session-only audio views; never persisted or reused across runs

def _window_records_of(k, chunk_s):
    from classroom_asr.backends import iter_silence_chunks
    key = (k, float(chunk_s))
    if key not in _WINDOW_RECORD_CACHE:
        wav = load_16k(interviews[k][0])
        _WINDOW_RECORD_CACHE[key] = [
            (start / 16000, (start + len(c)) / 16000, c)
            for start, c in iter_silence_chunks(wav, 16000, chunk_s) if len(c) >= 400
        ]
    return _WINDOW_RECORD_CACHE[key]

def _windows_of(k, chunk_s):
    return [c for _start, _end, c in _window_records_of(k, chunk_s)]

# Balance ALL ~chunk_s windows from ALL interviews evenly across the GPUs (so no GPU idles
# on the 2+1 interview split), transcribe, then reassemble each transcript in order. Output
# is identical to per-interview transcription (pure utilization win). `models` are preloaded
# (one per GPU) and NOT unloaded here (the caller owns them).
def window_pass(models, desc, *, chunk_s, batch_size):
    iv_records = [_window_records_of(k, chunk_s) for k in range(len(interviews))]
    tasks = [(k, wi, start, end, c) for k, rows in enumerate(iv_records)
             for wi, (start, end, c) in enumerate(rows)]
    shards = [tasks[i::len(models)] for i in range(len(models))]
    results = {}
    def worker(model, shard, pos):
        cks = [t[4] for t in shard]
        if not cks: return
        try:
            texts = model.transcribe_chunk_list(cks, batch_size=batch_size)
        except Exception as e:
            print(desc, "failed:", repr(e)[:120]); texts = [""] * len(cks)
        for (k, wi, _start, _end, _), txt in zip(shard, texts): results[(k, wi)] = txt
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(models))]
    for t in ts: t.start()
    for t in ts: t.join()
    out = []
    timed_parts = []
    for k, rows in enumerate(iv_records):
        parts = [results.get((k, wi), "") for wi in range(len(rows))]
        out.append(" ".join(p for p in parts if p).strip())
        timed_parts.append([
            {"start_s": round(start, 3), "end_s": round(end, 3), "text": parts[wi]}
            for wi, (start, end, _c) in enumerate(rows)
        ])
    WINDOW_PARTS[desc] = timed_parts
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

def error_counts_of(hyps):
    # RapidFuzz editops are one record per minimal edit, unlike opcodes whose records can span
    # several words. Keep S/D/I alongside aggregate WER so a seemingly better branch cannot hide
    # a deletion regression that matters to verbatim classroom transcription.
    S = D = I = R = 0
    for rt, h in zip(_reftok, hyps):
        R += len(rt)
        for tag, _src_pos, _dest_pos in Levenshtein.editops(
                rt, SCORE.tokens(h or "")).as_list():
            if tag == "replace": S += 1
            elif tag == "delete": D += 1
            elif tag == "insert": I += 1
    E = S + D + I
    return {"reference_words": R, "substitutions": S, "deletions": D, "insertions": I,
            "errors": E, "wer": round(E / R if R else 0.0, 4),
            "deletion_rate": round(D / R if R else 0.0, 4)}

def recall_floor(pool):
    # Fraction of REFERENCE words that NO branch produced anywhere. This is a RECALL lower bound
    # on achievable WER: it ignores insertions and is NOT a single realizable transcript (each ref
    # word may be recovered by a different branch). It answers "did the ensemble even hear this
    # word", not "what WER can a selector reach" -- for that honest ceiling see realizable_oracle_wer.
    unrec = R = 0
    for i, rt in enumerate(_reftok):
        hit = set()
        for hyps in pool:
            for tag, i0, i1, j0, j1 in Levenshtein.opcodes(rt, SCORE.tokens(hyps[i] or "")).as_list():
                if tag == "equal": hit.update(range(i0, i1))
        unrec += len(rt) - len(hit); R += len(rt)
    return unrec / R if R else 0.0

def realizable_oracle_wer(pool):
    # Honest ceiling: per interview build the confusion network from all branches, then pick the
    # per-slot candidate that best matches the reference -> a REAL transcript, scored with full WER
    # (insertions included). This is what a perfect selector over this candidate graph could reach.
    from classroom_asr.candidate_graph import build_graph as _bg, realizable_oracle_tokens as _roracle
    E = R = 0
    for i, rt in enumerate(_reftok):
        tls = [t for t in (SCORE.tokens(h[i] or "") for h in pool) if t]
        # ``pool`` is ordered with the primary Qwen transcript first.  Anchor the graph to that
        # transcript so the oracle measures the exact candidate graph seen by the selector.
        g = _bg(tls, pivot_index=0, opcodes_fn=fast_graph_opcodes)
        pivot = [s.pivot for s in g if s.kind == "word"]
        ops = Levenshtein.opcodes(pivot, rt).as_list()      # fast rapidfuzz pivot<->ref alignment
        E += Levenshtein.distance(rt, _roracle(g, rt, opcodes=ops)); R += len(rt)
    return E / R if R else 0.0

def add_branch(tag, hyps):
    # only count a branch if it actually produced text (a crashed branch is all "")
    got = sum(1 for h in hyps if h)
    if got == 0:
        print(f"[{tag:14s}] produced nothing (skipped from pool)"); return
    pool.append(hyps)
    print(f"[{tag:14s}] branch WER={wer_of(hyps):.3f}   recall_floor(pool)={recall_floor(pool):.3f}"
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

# Full large-v3 is not assumed to beat turbo: OpenAI reports dataset-dependent differences.
# Run it as a quality shadow with the documented beam-5 decode and FP16, then let this benchmark
# decide. It remains a separate candidate and does not silently replace the faster baseline.
hyp_A3 = None
if USE_WHISPER_LARGE_V3:
    hyp_A3 = run_windows("Whisper large-v3 quality", lambda dev: FasterWhisperASR(
        FW_QUALITY_MODEL, language="en", device=dev, compute_type="float16", beam_size=5,
        vad_filter=WHISPER_VAD), chunk_s=900, batch_size=1)
    add_branch("+WhisperLargeV3", hyp_A3)
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
                        lambda dev: Qwen3ASR(
                            QWEN3ASR_MODEL, language=None, device=dev,
                            max_new_tokens=QWEN_MAX_NEW_TOKENS),
                        chunk_s=QWEN_CHUNK_S, batch_size=8)
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
        t0 = time.time(); out = window_pass(
            vmodels, tag, chunk_s=VOXTRAL_CHUNK_S, batch_size=12)
        rec(tag, time.time() - t0); return out
    if USE_VOXTRAL:
        hyp_C = _vox_pass("transcription", "+Voxtral"); add_branch("+Voxtral", hyp_C)
    if USE_VOXTRAL_VERBATIM:
        hyp_VV = _vox_pass("verbatim", "+VoxtralVerbatim"); add_branch("+VoxtralVerbatim", hyp_VV)
    _free_models(vmodels); vmodels = None; del vmodels
"""),
    md("""## 9a. Branch CW / CWV — CrisperWhisper, independent + Qwen-conditioned
The verbatim lever: a Whisper fine-tune that *keeps* the `um`/`uh`/false starts the clean
branches delete. Its fast **CT2 runtime uses a forked `ctranslate2`** that can't share
site-packages with faster-whisper (branch A) — so CrisperWhisper runs in a
`--system-site-packages` **venv** (reuses the main torch/transformers, shadows only
`ctranslate2` with the fork) driven by a subprocess. The venv was built in the background
back in §2a, so this cell usually just joins it and transcribes. Branch A keeps stock CT2;
the main kernel never imports the fork. **Both GPUs are used**: one pinned subprocess per GPU
transcribes a disjoint slice of the interviews in parallel (whole files, so each keeps
CrisperWhisper's native chunking and the output is identical).

The same loaded model also runs **Verbatimize** on each matching Qwen 30-second audio window.
That path treats Qwen as the content transcript and asks Crisper to insert only acoustically
supported fillers, repetitions, and repairs. Crisper documents Verbatimize for audio no longer
than 30 seconds, so we deliberately use the exact Qwen window/audio pairs rather than a whole
interview. Both outputs are scored independently. The worker prints which backend it got —
`ct2 (fast)` or the `transformers (SLOW)` fallback — so a slow run is visible."""),
    code(r"""
hyp_CW = hyp_CWV = None
if USE_CRISPER:
    import os, sys, json, subprocess, threading, tempfile
    _cw_t0 = time.time()
    # Worker source and audio-derived handoff JSON are session-temporary. Only the venv and
    # converted CT2 model belong in the Files-only persisted CW_WORK directory.
    CW_RUN = tempfile.mkdtemp(prefix="classroom_asr_cw_")
    WORKER = os.path.join(CW_RUN, "cw_worker.py")
    try:
        if _cw["thread"] is not None:
            _cw["thread"].join()                 # wait for the background venv build (§2a)
        if _cw["err"] is not None:
            raise _cw["err"]
        with open(WORKER, "w") as f:
            f.write(r'''
import sys, os, json, warnings, logging, fcntl, hashlib, time
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
size, cache_dir, inp, outp, init_lock = sys.argv[1:6]
payload = json.load(open(inp))
paths = payload["paths"]
qwen_windows = payload.get("qwen_windows", {})

def _official_model(name):
    return name if "/" in name else f"nyralabs/CrisperWhisper2.0_{name}"

def _cached_model_arg(name):
    # CrisperWhisper normally resolves the HF source before consulting its conversion cache.
    # Passing the completed local CT2 directory avoids re-downloading source weights next session.
    repo = _official_model(name)
    slug = repo.replace("/", "--").replace("\\", "--")
    key = f"{slug}_float16_{hashlib.sha256(repo.encode()).hexdigest()[:12]}"
    candidate = os.path.join(cache_dir, key)
    if (os.path.isfile(os.path.join(candidate, "model.bin"))
            and os.path.isfile(os.path.join(candidate, ".conversion_complete"))):
        return candidate
    return name
# The first CT2 load converts the HF model into a shared local cache. Serialize that conversion so
# a second worker cannot observe half-written files. A completed persisted conversion is immutable,
# so both isolated GPU processes can load it concurrently on later runs instead of serializing two
# ordinary model loads behind this lock.
try:
    main_arg = _cached_model_arg(size)
    if main_arg == size:
        with open(init_lock, "w") as _lock_file:
            fcntl.flock(_lock_file, fcntl.LOCK_EX)
            try:
                # Worker 0 may have completed conversion while this worker waited.
                main_arg = _cached_model_arg(size)
                m = CrisperWhisperModel(main_arg, backend="ct2", cache_dir=cache_dir)
            finally:
                fcntl.flock(_lock_file, fcntl.LOCK_UN)
    else:
        m = CrisperWhisperModel(main_arg, backend="ct2", cache_dir=cache_dir)
    print("CW backend: ct2 (fast)", flush=True)
except Exception as e:
    print("CW backend: transformers (SLOW) -- ct2 failed:", repr(e)[:200], flush=True)
    m = CrisperWhisperModel(size, backend="transformers")
t0 = time.perf_counter(); independent = {}
for p in paths:
    try:
        r = m.transcribe(p, language="en"); independent[p] = getattr(r, "text", "") or ""
    except Exception as e:
        independent[p] = ""; print("independent fail", p, repr(e)[:120], file=sys.stderr)
independent_s = time.perf_counter() - t0

# Verbatimize is explicitly a <=30 s API. Reuse the full recording once, slice it by the
# silence-snapped Qwen timestamps, and pair every slice with exactly the Qwen text decoded from
# that slice. Inputs and outputs live only in this session-temporary worker handoff.
t0 = time.perf_counter(); verbatized = {}
if qwen_windows:
    from crisperwhisper.audio import load_audio
    for p in paths:
        rows = qwen_windows.get(p, [])
        if not rows:
            continue
        audio = load_audio(p)
        parts = []
        for wi, row in enumerate(rows):
            transcript = (row.get("text") or "").strip()
            if not transcript:
                parts.append(""); continue
            start = max(0, int(round(float(row["start_s"]) * 16000)))
            end = min(len(audio), int(round(float(row["end_s"]) * 16000)))
            try:
                r = m.verbatimize(audio[start:end], transcript, language="en", sr=16000)
                parts.append(getattr(r, "text", "") or "")
            except Exception as e:
                parts.append("")
                print("verbatize fail", p, wi, repr(e)[:120], file=sys.stderr)
        verbatized[p] = parts
verbatize_s = time.perf_counter() - t0
json.dump({"independent": independent, "verbatized": verbatized,
           "timings": {"independent_s": independent_s, "verbatize_s": verbatize_s}},
          open(outp, "w"))
''')
        paths = [os.path.abspath(str(interviews[k][0])) for k in range(len(interviews))]
        # One worker per GPU, pinned via CUDA_VISIBLE_DEVICES, run in parallel so both GPUs
        # transcribe instead of one sitting idle. Whole files are split across GPUs (no
        # cross-file interaction and each file keeps CrisperWhisper's native internal 30s
        # chunking), so the per-file output is byte-identical to the single-worker run.
        gpus = [g for g in GPUS if g is not None] or [None]
        # Keep each interview whole (identical decoding), but use longest-processing-time-first
        # assignment instead of the old 2+1 round robin. This minimizes idle tail time whenever
        # interview lengths differ, without using transcript/reference information.
        duration_s = {
            path: len(load_16k(interviews[k][0])) / 16000.0
            for k, path in enumerate(paths)
        }
        shards = [[] for _ in gpus]; shard_seconds = [0.0 for _ in gpus]
        for path in sorted(paths, key=lambda p: duration_s[p], reverse=True):
            gi = min(range(len(gpus)), key=lambda i: (shard_seconds[i], i))
            shards[gi].append(path); shard_seconds[gi] += duration_s[path]
        print("Crisper whole-file GPU plan:",
              [f"GPU{g}: {len(shards[i])} files, {shard_seconds[i]/60:.1f} min"
               for i, g in enumerate(gpus)], flush=True)
        res = {}; verb_res = {}; worker_timings = []
        _qwen_parts = WINDOW_PARTS.get("A+B+Qwen3", [[] for _ in interviews])
        qwen_windows_by_path = {
            paths[k]: _qwen_parts[k] for k in range(len(paths))
        } if USE_CRISPER_VERBATIZE and hyp_Z and any(hyp_Z) else {}
        init_lock = os.path.join(CW_RUN, "ct2_model_init.lock")
        def _cw_run(gi, gpu, shard):
            if not shard:
                return
            try:
                inp = os.path.join(CW_RUN, f"in_{gi}.json"); outp = os.path.join(CW_RUN, f"out_{gi}.json")
                json.dump({"paths": shard,
                           "qwen_windows": {p: qwen_windows_by_path.get(p, []) for p in shard}},
                          open(inp, "w"))
                env = dict(os.environ)
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # each subprocess sees exactly one GPU
                subprocess.run([CW_VENV_PY, WORKER, CRISPER_SIZE, CW_CT2_CACHE,
                                inp, outp, init_lock],
                               check=True, env=env)
                worker_result = json.load(open(outp))
                res.update(worker_result.get("independent", {}))
                verb_res.update(worker_result.get("verbatized", {}))
                worker_timings.append(worker_result.get("timings", {}))
            except Exception as e:                          # don't crash the thread; skip cleanly
                print(f"CrisperWhisper GPU{gpu} worker failed: {repr(e)[:160]}", flush=True)
        _cw_ts = [threading.Thread(target=_cw_run, args=(gi, gpus[gi], shards[gi]))
                  for gi in range(len(gpus))]
        for t in _cw_ts: t.start()
        for t in _cw_ts: t.join()
        from classroom_asr.backends.crisperwhisper_asr import CrisperWhisperV2
        hyp_CW = [CrisperWhisperV2._clean(res.get(p, "")) for p in paths]   # [um]->um, drop [laughter]
        if qwen_windows_by_path:
            hyp_CWV = [" ".join(
                CrisperWhisperV2._clean(part) for part in verb_res.get(p, []) if part
            ).strip() for p in paths]
        if worker_timings:
            print("Crisper worker phase maxima:",
                  {key: round(max(float(t.get(key, 0.0)) for t in worker_timings), 1)
                   for key in ("independent_s", "verbatize_s")}, flush=True)
    except Exception as e:
        print("CrisperWhisper isolation failed:", repr(e)[:200])
        hyp_CW = hyp_CWV = [""] * len(interviews)
    rec("+CrisperWhisper", time.time() - _cw_t0)
    add_branch("+CrisperWhisper", hyp_CW)
    if hyp_CWV and any(hyp_CWV): add_branch("+CrisperQwenVerbatize", hyp_CWV)
"""),
    md("""## 10. Phone branches — timestamped acoustic evidence for the LLM (paused with selector)
The phone path is *pronunciation evidence*, not a word transcript. CORAAL has no phonetic
reference, so PER/IPA-CER cannot be reported here and IPA is never forced into word WER.
However, it is no longer diagnostic-only: both phone streams are kept in timestamped ~24 s
windows and the selector retrieves the windows overlapping each uncertain word region (§10,
§14.4, §15.4). Repeated evidence blocks are deduplicated inside each selector batch.

Two independent phone candidates are preserved: `wav2vec2-lv-60-espeak` (manual vocab CTC
decoder for transformers 4.57.x compatibility) and **PhoneticXeus**, the design's default
accented-English/multilingual phone recognizer. The backends currently expose one path each;
true within-model N-best/posterior lattices and robust P2G remain the next upstream milestone.
While `USE_LLM_SELECTOR=False`, both phone flags follow it and this section performs no model
download or inference; the implementation remains ready for the selector's return."""),
    code(r"""
phone_evidence = [[] for _ in interviews]

def phone_window_pass(desc, make_model, *, batch_size):
    # Timestamped PhonePath records, balanced by ~24 s window across all GPUs.
    t0 = time.time()
    try:
        models = load_models(make_model)
    except Exception as e:
        print(f"[{desc}] load FAILED: {repr(e)[:160]}"); _free(); return [[] for _ in interviews]
    tload = time.time() - t0
    records = [_window_records_of(k, 24.0) for k in range(len(interviews))]
    tasks = [(k, wi, start, end, c) for k, rows in enumerate(records)
             for wi, (start, end, c) in enumerate(rows)]
    shards = [tasks[i::len(models)] for i in range(len(models))]
    results = {}
    def worker(model, shard, pos):
        i, bs = 0, batch_size
        pbar = tqdm(total=len(shard), desc=f"{desc}:{pos}", position=pos, leave=True)
        while i < len(shard):
            batch = shard[i:i + bs]
            try:
                paths = model.recognize_batch([row[4] for row in batch], top_k=3)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    torch.cuda.empty_cache(); bs = max(1, bs // 2); continue
                raise
            for (k, wi, _start, _end, _c), found in zip(batch, paths):
                results[(k, wi)] = [
                    {"id": f"{desc}:{p.id}", "source": desc, "ipa": p.ipa, "p": float(p.prob)}
                    for p in found if p.ipa
                ]
            i += len(batch); pbar.update(len(batch))
        pbar.close()
    try:
        t1 = time.time()
        ts = [threading.Thread(target=worker, args=(models[i], shards[i], i))
              for i in range(len(models))]
        for t in ts: t.start()
        for t in ts: t.join()
        out = [[{"start_s": round(start, 3), "end_s": round(end, 3),
                 "paths": results.get((k, wi), [])}
                for wi, (start, end, _c) in enumerate(rows)]
               for k, rows in enumerate(records)]
        print(f"   {desc}: load {tload:.1f}s + transcribe {time.time()-t1:.1f}s", flush=True)
        return out
    finally:
        _free_models(models)
        rec(desc, time.time() - t0)

_phone_sources = []
if USE_PHONE:
    try:
        from classroom_asr.backends.wav2vec2_phone import Wav2Vec2Phone
        _phone_sources.append(phone_window_pass(
            "wav2vec2-phone", lambda dev: Wav2Vec2Phone(PHONE_MODEL, device=dev), batch_size=8))
    except Exception as e:
        print("phone branch skipped:", repr(e)[:200])

if USE_PHONETIC_XEUS:
    try:
        from classroom_asr.backends.phonetic_xeus import PhoneticXeus
        _phone_sources.append(phone_window_pass(
            "PhoneticXeus", lambda dev: PhoneticXeus(PHONETIC_XEUS_MODEL, device=dev), batch_size=1))
    except Exception as e:
        print("PhoneticXeus skipped:", repr(e)[:200])

# Both sources use the same deterministic silence-snapped 24 s boundaries. Merge their paths
# into one immutable evidence record per time window; a failed source simply contributes none.
for k in range(len(interviews)):
    merged = {}
    for source_rows in _phone_sources:
        for row in source_rows[k]:
            key = (row["start_s"], row["end_s"])
            merged.setdefault(key, []).extend(row["paths"])
    phone_evidence[k] = [
        {"start_s": start, "end_s": end, "paths": paths}
        for (start, end), paths in sorted(merged.items()) if paths
    ]
print(f"phone evidence windows: {sum(len(rows) for rows in phone_evidence)}; "
      f"paths: {sum(len(row['paths']) for rows in phone_evidence for row in rows)}")
if phone_evidence and phone_evidence[0]:
    print("retrievable phone evidence example:", phone_evidence[0][0])
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
    md("""## 11.5 Recall-floor breakdown — words the *whole ensemble* never produced
Section 11 is branch A alone. This is the **candidate recall floor**: reference words that **no
branch** produced anywhere (a word counts as recovered iff it lands in an `equal` span for some
branch). Note this is a **recall lower bound**, *not* an achievable WER — it ignores insertions
and isn't a single realizable transcript; the honest achievable ceiling is `realizable_oracle_wer`
in §11.6. Still, it's exactly the set no selector can ever recover, so its make-up is the useful
signal here.

Vernacular is deliberately **kept as errors** (CORAAL is regional AAL — `gonna`, `y'all`,
g-dropping are the features of interest, not noise to fold away), so they surface here as
genuine misses. The category split separates convention/vernacular from the model-limited
deletions (backchannel overlap, reduced function words)."""),
    code(r"""
from collections import Counter
# Same "recovered" rule as recall_floor(): a ref index is recovered iff some branch aligns it
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
    print(f"CANDIDATE RECALL FLOOR: {total_unrec} of {R} ref words produced by NO branch"
          f"   (unrecovered-word rate {total_unrec/R:.3f}; a recall lower bound, not achievable WER)")
    print("\nby category (share of the floor):")
    for c, n in bycat.most_common():
        print(f"   {n:5d}  {100*n/total_unrec:4.1f}%   {c}")
    print("\nMost-common unrecovered reference words (what the whole ensemble misses):")
    for w, n in floor.most_common(30):
        print(f"   {w!r:16} x{n:<4} [{cat(w)}]")
"""),
    md("""## 11.6 Selection — Qwen-anchored candidate graph
The **primary backbone** is Qwen3-ASR, as specified by the design. Other word recognizers supply
alternatives at aligned word and insertion slots; they do not vote a separate consensus
transcript into existence. The Qwen backbone therefore remains intact wherever the selector is
not asked, abstains, or fails.

The **realizable oracle** is the best WER a perfect selector could reach *over this same
Qwen-anchored candidate graph* — a real assembled transcript scored with full WER (insertions
included), so it is an honest ceiling unlike the recall floor (§11.5). Agreement between branches
is still useful as an uncertainty signal, but majority-vote fusion is no longer a transcript
variant or an acceptance target."""),
    code(r"""
from classroom_asr.candidate_graph import build_graph
from classroom_asr.selector import build_decisions
# Qwen is deliberately first: build_graph(..., pivot_index=0) and every selector fallback are
# anchored to it. Phone branches are IPA evidence, not word candidates.
_wb_named = [("Qwen3-ASR", globals().get("hyp_Z")),
             ("Whisper", globals().get("hyp_A")),
             ("WhisperLargeV3", globals().get("hyp_A3")),
             ("wav2vec2 CTC", globals().get("hyp_B")),
             ("Voxtral", globals().get("hyp_C")),
             ("VoxtralVerbatim", globals().get("hyp_VV")),
             ("CrisperWhisper", globals().get("hyp_CW")),
             ("CrisperQwenVerbatize", globals().get("hyp_CWV"))]
_wb_named = [(name, h) for name, h in _wb_named if h and any(h)]
if not _wb_named:
    raise RuntimeError("no word-recognition branch produced a candidate transcript")
backbone_name, backbone_hyps = _wb_named[0]
if backbone_name != "Qwen3-ASR":
    print(f"Qwen3-ASR unavailable; emergency backbone is {backbone_name}")
_wb = [h for _, h in _wb_named]
canonical = list(backbone_hyps)
canonical_source = backbone_name
canonical_wer = wer_of(canonical)
r_oracle = realizable_oracle_wer(_wb)
_decisions = _total = 0
for k in range(len(interviews)):
    graph = build_graph([SCORE.tokens(h[k]) for h in _wb if h[k]], pivot_index=0,
                        opcodes_fn=fast_graph_opcodes)
    _total += len(graph); _decisions += len(build_decisions(graph))
print(f"Qwen-anchored graph from {len(_wb)} word branches: "
      + ", ".join(name for name, _ in _wb_named))
print(f"[BACKBONE       ] {backbone_name} WER={canonical_wer:.3f}"
      f"   vs baseline A={wer_of(hyp_A):.3f}  vs realizable oracle={r_oracle:.3f}"
      f"  (recall floor={recall_floor(pool):.3f})")
print(f"selector decisions: {_decisions}/{_total} aligned slots "
      f"({100*_decisions/max(_total,1):.1f}%); every unselected slot keeps the backbone token")

# Exact leave-one-branch-out analysis for both the candidate recall floor and the realizable
# Qwen-anchored oracle. Each branch is
# aligned to the reference once with RapidFuzz; a reference occurrence is "unique" when this is
# the only word branch that recovered it. Removing that branch raises the recall floor by exactly
# unique/R. We also rebuild the inexpensive text candidate graph without each branch and score its
# exact realizable oracle, so a low unique count cannot hide useful candidate placement.
_R = sum(len(rt) for rt in _reftok)
_branch_hits = []
for _name, _hyps in _wb_named:
    _hits = set()
    for _k, (_rt, _hyp) in enumerate(zip(_reftok, _hyps)):
        for _tag, _i0, _i1, _j0, _j1 in Levenshtein.opcodes(
                _rt, SCORE.tokens(_hyp or "")).as_list():
            if _tag == "equal":
                _hits.update((_k, _ri) for _ri in range(_i0, _i1))
    _branch_hits.append(_hits)
_all_hits = set().union(*_branch_hits) if _branch_hits else set()
_timing_by_name = {name: dt for name, dt in TIMINGS}
_stage_for_branch = {"Qwen3-ASR": "A+B+Qwen3", "Whisper": "A whisper",
                     "WhisperLargeV3": "Whisper large-v3 quality",
                     "wav2vec2 CTC": "A+B", "Voxtral": "+Voxtral",
                     "VoxtralVerbatim": "+VoxtralVerbatim",
                     "CrisperWhisper": "+CrisperWhisper",
                     "CrisperQwenVerbatize": "+CrisperWhisper"}
branch_overlap_ablation = []
print("\n=== word-branch overlap: exact leave-one-out floor + graph-oracle effect ===")
print("branch                 WER   unique  floor_without  oracle_without  oracle_delta  stage_s")
for _bi, ((_name, _hyps), _hits) in enumerate(zip(_wb_named, _branch_hits)):
    _other_hits = set().union(*(_branch_hits[:_bi] + _branch_hits[_bi + 1:]))
    _unique = _hits - _other_hits
    _overlap = len(_hits & _other_hits) / max(len(_hits), 1)
    _floor_without = (_R - len(_other_hits)) / max(_R, 1)
    _other_branches = _wb[:_bi] + _wb[_bi + 1:]
    _oracle_without = realizable_oracle_wer(_other_branches)
    _oracle_delta = _oracle_without - r_oracle
    _stage_s = _timing_by_name.get(_stage_for_branch.get(_name, ""))
    _row = {"branch": _name, "wer": round(wer_of(_hyps), 4),
            "reference_hits": len(_hits), "unique_reference_hits": len(_unique),
            "hit_overlap_fraction": round(_overlap, 4),
            "recall_floor_without": round(_floor_without, 4),
            "recall_floor_increase_if_removed": round(len(_unique) / max(_R, 1), 4),
            "realizable_oracle_without": round(_oracle_without, 4),
            "realizable_oracle_increase_if_removed": round(_oracle_delta, 4),
            "stage_seconds": round(_stage_s, 1) if _stage_s is not None else None}
    branch_overlap_ablation.append(_row)
    print(f"{_name:22s} {_row['wer']:.3f} {len(_unique):7d} "
          f"{_floor_without:14.3f} {_oracle_without:15.3f} {_oracle_delta:+12.3f} "
          f"{_stage_s if _stage_s is not None else float('nan'):8.1f}")
print(f"full-pool recall floor: {(_R-len(_all_hits))/max(_R,1):.3f}")
branch_pair_overlap = []
print("\n=== pairwise overlap of correctly recovered reference occurrences ===")
print("branch pair                                      shared  smaller-covered")
for _left in range(len(_wb_named)):
    for _right in range(_left + 1, len(_wb_named)):
        _shared = len(_branch_hits[_left] & _branch_hits[_right])
        _smaller_covered = _shared / max(
            min(len(_branch_hits[_left]), len(_branch_hits[_right])), 1)
        _pair = {"left": _wb_named[_left][0], "right": _wb_named[_right][0],
                 "shared_reference_hits": _shared,
                 "smaller_hit_set_covered_fraction": round(_smaller_covered, 4)}
        branch_pair_overlap.append(_pair)
        print(f"{_pair['left'] + ' / ' + _pair['right']:48s} {_shared:7d} "
              f"{100*_smaller_covered:14.1f}%")
print("A zero/small unique count identifies overlap worth investigating; it is not by itself "
      "authorization to remove a branch because oracle placement and selector evidence may differ.")

# Exhaustive Qwen-anchored subset frontier. With at most seven optional word branches this is
# only 2^7=128 inexpensive text-graph evaluations and requires no ASR rerun. Runtime accounting
# deduplicates shared stages: either Voxtral mode pays the shared load once; either Crisper output
# pays the combined worker once. Keep a subset only when no cheaper/equal-cost subset has an equal
# or lower realizable-oracle WER.
from itertools import combinations
_stage_groups = {
    "Qwen3-ASR": ("A+B+Qwen3",),
    "Whisper": ("A whisper",),
    "WhisperLargeV3": ("Whisper large-v3 quality",),
    "wav2vec2 CTC": ("A+B",),
    "Voxtral": ("Voxtral load (shared)", "+Voxtral"),
    "VoxtralVerbatim": ("Voxtral load (shared)", "+VoxtralVerbatim"),
    "CrisperWhisper": ("+CrisperWhisper",),
    "CrisperQwenVerbatize": ("+CrisperWhisper",),
}
_timing_by_stage = {name: dt for name, dt in TIMINGS}
branch_subset_results = []
_optional_indices = list(range(1, len(_wb_named)))  # Qwen at index 0 is always the pivot
for _count in range(len(_optional_indices) + 1):
    for _chosen_optional in combinations(_optional_indices, _count):
        _chosen = (0,) + _chosen_optional
        _names = [_wb_named[i][0] for i in _chosen]
        _subset_pool = [_wb[i] for i in _chosen]
        _subset_hits = set().union(*(_branch_hits[i] for i in _chosen))
        _stages = {stage for name in _names for stage in _stage_groups.get(name, ())}
        _cost = sum(_timing_by_stage.get(stage, 0.0) for stage in _stages)
        branch_subset_results.append({
            "branches": _names,
            "branch_count": len(_names),
            "realizable_oracle_wer": round(realizable_oracle_wer(_subset_pool), 4),
            "recall_floor": round((_R - len(_subset_hits)) / max(_R, 1), 4),
            "estimated_stage_seconds": round(_cost, 1),
        })

branch_subset_pareto = []
for _candidate in sorted(branch_subset_results,
                         key=lambda row: (row["estimated_stage_seconds"],
                                          row["realizable_oracle_wer"], row["branch_count"])):
    _dominated = any(
        other["estimated_stage_seconds"] <= _candidate["estimated_stage_seconds"]
        and other["realizable_oracle_wer"] <= _candidate["realizable_oracle_wer"]
        and (other["estimated_stage_seconds"] < _candidate["estimated_stage_seconds"]
             or other["realizable_oracle_wer"] < _candidate["realizable_oracle_wer"])
        for other in branch_subset_results
    )
    if not _dominated:
        branch_subset_pareto.append(_candidate)
print("\n=== Qwen-anchored accuracy/runtime Pareto frontier (all branch subsets) ===")
print("stage_s  oracle  floor  branches")
for _row in branch_subset_pareto:
    print(f"{_row['estimated_stage_seconds']:7.1f}  {_row['realizable_oracle_wer']:.3f}  "
          f"{_row['recall_floor']:.3f}  {', '.join(_row['branches'])}")
"""),
    md("""## 11.7 Acoustic-evidence-aware local judge (Qwen3.5-9B) — currently paused
**Scope, honestly:** this is still not the complete §14–15 whole-lesson selector. It receives
local word context plus the timestamp-overlapping wav2vec2/PhoneticXEUS realized-IPA paths for
each contested region. Evidence blocks are retrieved through a Qwen/Voxtral/Whisper timestamped
text anchor and deduplicated per prompt batch. CORAAL's single mixed-speaker recording does not
provide the production system's separate teacher/student channels, so role, overlap, lesson
vocabulary, phonetic RAG, and robust P2G are still absent here.

The judge picks only among branch candidate IDs (`needs_novel_candidate`/NEW off, §14.5). Instead
of generating prose and parsing `N:LETTER`, Qwen3.5 directly scores only the valid next-token
candidate IDs for each span, batched across decisions. This is both structurally constrained and
avoids hundreds of generated answer tokens. It runs in its own venv at full fp16 across both freed
T4s (§21).

On a successful run, the constrained selector result is the canonical transcript. If selection
fails, the canonical result remains the Qwen backbone. The same forward-pass scores are also
reassembled at several non-default logit-margin thresholds for diagnostics only; references never
choose the canonical threshold or transcript."""),
    code(r"""
if USE_LLM_SELECTOR:
    import os, json, subprocess, tempfile, time
    _sel_t0 = time.time()
    # Keep only the reusable venv in SEL_WORK. Transcripts, phone evidence, decisions, worker
    # source, and selected output are derived per run and live in the ephemeral session temp dir.
    SEL_RUN = tempfile.mkdtemp(prefix="classroom_asr_selector_")
    SEL_WORKER = os.path.join(SEL_RUN, "sel_worker.py")
    llm_selected = None; llm_selected_wer = None; selector_stats = None
    selector_threshold_wer = None
    try:
        if _sel.get("thread") is not None: _sel["thread"].join()   # wait for the venv build (§2a)
        if _sel.get("err") is not None: raise _sel["err"]
        with open(SEL_WORKER, "w") as f:
            # Raw literal: the embedded worker contains regexes and ``\n`` string escapes that
            # must reach its source unchanged instead of being decoded by this notebook cell.
            f.write(r'''
import sys, json, math, torch, transformers
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForMultimodalLM, AutoProcessor
from phonemizer import phonemize
import classroom_asr.selector as selector_module
from classroom_asr.phonetics import best_phone_subsequence
from classroom_asr.selector import (
    build_decisions, format_batch, select_graph_with_chooser,
)
from classroom_asr.normalize import Normalizer
SCORE = Normalizer(fold_numbers=True, fold_spelling=True)
model_id, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(inp))

def _evidence_by_slot(branches, anchor_chunks, phone_chunks):
    # Map Qwen-anchored candidate slots to timestamped phone windows through a fast text-anchor
    # alignment, then reduce each raw phone window to candidate-local pronunciation matches.
    graph = selector_module.build_graph(
        [SCORE.tokens(text) for text in branches if text], pivot_index=0,
        opcodes_fn=lambda pivot, hyp: Levenshtein.opcodes(pivot, hyp).as_list())
    pivot = [slot.pivot for slot in graph if slot.kind == "word"]
    anchor, anchor_chunk = [], []
    for ci, chunk in enumerate(anchor_chunks or []):
        tokens = SCORE.tokens(chunk.get("text", ""))
        anchor.extend(tokens); anchor_chunk.extend([ci] * len(tokens))
    if not pivot or not anchor or not phone_chunks:
        return {}, graph

    # Every pivot word gets an anchor-token position. Equal/replace blocks map proportionally;
    # deleted pivot words inherit the nearest boundary. This is only evidence retrieval, never
    # reference scoring, and uses no CORAAL transcript boundaries or gold text.
    pivot_to_anchor = {}
    for tag, i0, i1, j0, j1 in Levenshtein.opcodes(pivot, anchor).as_list():
        pn, an = i1 - i0, j1 - j0
        if tag in ("equal", "replace") and an:
            for off in range(pn):
                pivot_to_anchor[i0 + off] = j0 + min(an - 1, (off * an) // max(pn, 1))
        elif tag == "delete":
            nearest = min(max(j0, 0), len(anchor) - 1)
            for pi in range(i0, i1): pivot_to_anchor[pi] = nearest

    decisions = build_decisions(graph)
    paths_by_slot = {}
    raw_chars = 0
    for decision in decisions:
        slot = graph[decision.slot]
        pi = ((decision.slot - 1) // 2 if slot.kind == "word" else
              min(decision.slot // 2, len(pivot) - 1))
        ai = pivot_to_anchor.get(pi)
        if ai is None or not (0 <= ai < len(anchor_chunk)): continue
        chunk = anchor_chunks[anchor_chunk[ai]]
        start, end = float(chunk["start_s"]), float(chunk["end_s"])
        paths = []
        for phone_chunk in phone_chunks:
            if float(phone_chunk["end_s"]) <= start or float(phone_chunk["start_s"]) >= end:
                continue
            for path in phone_chunk.get("paths", []):
                ipa = (path.get("ipa") or "").strip()
                if ipa:
                    raw_chars += len(ipa)
                    paths.append({"source": path.get("source", "phone"),
                                  "id": path.get("id", "p?"),
                                  "p": float(path.get("p", 0.0)), "ipa": ipa})
        if paths:
            paths_by_slot[decision.slot] = paths

    # G2P every distinct candidate-in-context once. The selector never receives the full raw
    # 24-second phone stream: it sees only the best local realized excerpt for each advertised
    # candidate, plus an accent-aware phone similarity score.
    phrase_by_choice = {}
    phrases = set()
    for decision in decisions:
        for letter, token in decision.options:
            if token is selector_module.NULL:
                continue
            phrase = " ".join([*decision.before[-1:], str(token), *decision.after[:1]])
            phrase_by_choice[(decision.slot, letter)] = phrase
            phrases.add(phrase)
    ordered_phrases = sorted(phrases)
    expected = phonemize(
        ordered_phrases, language="en-us", backend="espeak", strip=True,
        preserve_punctuation=False, with_stress=True, njobs=1)
    if isinstance(expected, str):
        expected = [expected]
    ipa_by_phrase = dict(zip(ordered_phrases, expected))

    evidence = {}
    compact_chars = 0
    for decision in decisions:
        paths = paths_by_slot.get(decision.slot, [])
        if not paths:
            continue
        lines = []
        for letter, token in decision.options:
            if token is selector_module.NULL:
                continue
            phrase = phrase_by_choice[(decision.slot, letter)]
            expected_ipa = ipa_by_phrase.get(phrase, "")
            best_by_source = {}
            for path in paths:
                similarity, excerpt = best_phone_subsequence(expected_ipa, path["ipa"], flank=4)
                prior = best_by_source.get(path["source"])
                if prior is None or similarity > prior[0]:
                    best_by_source[path["source"]] = (similarity, excerpt, path)
            matches = []
            best_excerpt = ""
            best_similarity = -1.0
            for source, (similarity, excerpt, path) in sorted(best_by_source.items()):
                matches.append(f"{source}={similarity:.3f}(p={path['p']:.3f})")
                if similarity > best_similarity:
                    best_similarity, best_excerpt = similarity, excerpt
            if matches:
                line = (f"{letter}={token} exp/{expected_ipa}/ " + " ".join(matches)
                        + f" heard≈/{best_excerpt}/")
                compact_chars += len(line)
                lines.append(line)
        if lines:
            evidence[decision.slot] = lines
    print(f"selector compact phone evidence: {len(evidence)}/{len(decisions)} decisions; "
          f"raw-window IPA chars={raw_chars} candidate-local chars={compact_chars}",
          file=sys.stderr, flush=True)
    return evidence, graph

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"                 # final position is the answer position for all rows
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
# Full fp16, sharded across BOTH T4s (acoustic models are unloaded -> 32 GB free; 9B fp16 ~18 GB).
# No quantization: the design wants the judge validated at full precision first (§21). fp16 (not
# bf16) because T4/Turing has fp16 tensor cores but no bf16 path.
try:
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto").eval()
except TypeError:
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto").eval()
_dev = next(model.parameters()).device            # feed inputs to the shard holding the embeddings
print(f"selector runtime: transformers={transformers.__version__} model={type(model).__name__}",
      file=sys.stderr)
print(f"selector package: {selector_module.__file__}", file=sys.stderr)

# Candidate IDs are deliberately single ASCII letters. Derive their IDs after the exact rendered
# assistant header rather than assuming standalone tokenization matches the causal answer position.
_letter_token = {}
_token_probe_messages = [{"role": "user", "content": [
    {"type": "text", "text": "Answer with one candidate letter.\nCandidate ID:"}]}]
_token_probe_text = processor.apply_chat_template(
    _token_probe_messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
_token_probe_prefix = tokenizer.encode(_token_probe_text, add_special_tokens=False)
for _letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _with_answer = tokenizer.encode(_token_probe_text + _letter, add_special_tokens=False)
    if (_with_answer[:len(_token_probe_prefix)] != _token_probe_prefix
            or len(_with_answer) != len(_token_probe_prefix) + 1):
        raise RuntimeError(f"selector candidate ID {_letter!r} is not one contextual token: "
                           f"prefix={_token_probe_prefix[-4:]} answer={_with_answer[-5:]}")
    _letter_token[_letter] = _with_answer[-1]

_stats = {"prefill_attempts": 0, "forward_batches": 0, "decode_steps": 0,
          "prompt_tokens": 0, "completed_decisions": 0,
          "oom_splits": 0, "backbone_overrides": 0,
          "evidenced_backbone_overrides": 0, "margins": []}
_decision_scores = {}
_debugged = False
_total_decisions = 0

def _score_once(decisions):
    # Put the whole batch in one conversation so decisions that share a 24-second phone window
    # also share one IPA block in the prompt. The old one-conversation-per-decision layout repeated
    # that large block up to 24 times and could spend many minutes on the very first forward pass.
    prompt = format_batch(decisions) + "\n1:"
    conversations = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, enable_thinking=False, tokenize=True,
        return_dict=True, return_tensors="pt").to(_dev)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    _stats["prefill_attempts"] += 1
    attempt_number = _stats["prefill_attempts"]
    print(f"selector attempt {attempt_number}: prefill {len(decisions)} decisions in "
          f"{prompt_tokens} shared prompt tokens", file=sys.stderr, flush=True)
    with torch.inference_mode():
        # Prefill the shared evidence once, then keep its KV cache while emitting only the chosen
        # candidate ID and the deterministic next-item scaffold. Transcript text is never freely
        # generated and only advertised letter logits are eligible at each answer position.
        outputs = model(**inputs, logits_to_keep=1, use_cache=True)
        logits = outputs.logits[0, -1, :].float()
        cache = outputs.past_key_values
        attention_mask = inputs["attention_mask"]
    _stats["forward_batches"] += 1
    _stats["prompt_tokens"] += prompt_tokens
    choices = {}
    global _debugged
    for row, decision in enumerate(decisions):
        scored = [(letter, token, float(logits[_letter_token[letter]].item()))
                  for letter, token in decision.options]
        scored.sort(key=lambda item: item[2], reverse=True)
        best_letter, best_token, best_score = scored[0]
        margin = best_score - scored[1][2] if len(scored) > 1 else math.inf
        _stats["margins"].append(margin)
        if best_token != decision.default:
            _stats["backbone_overrides"] += 1
            if decision.evidence:
                _stats["evidenced_backbone_overrides"] += 1
        _decision_scores[decision.slot] = (best_token, margin)
        choices[decision.slot] = best_token
        if not _debugged:
            _debugged = True
            print("=== SAMPLE SHARED CONSTRAINED SELECTOR PROMPT (head) ===", file=sys.stderr)
            print(prompt[:1200], file=sys.stderr)
            print("=== SAMPLE RESTRICTED CANDIDATE SCORES ===", file=sys.stderr)
            print([(letter, str(token), round(score, 4)) for letter, token, score in scored],
                  f"chosen={best_letter} margin={margin:.4f}", file=sys.stderr)
        if row + 1 < len(decisions):
            # The answer letter is selected above; the rest is fixed syntax, not model output.
            suffix = best_letter + f"\n{row + 2}:"
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            if not suffix_ids or suffix_ids[0] != _letter_token[best_letter]:
                raise RuntimeError(f"selector scaffold tokenization changed for {suffix!r}: "
                                   f"{suffix_ids[:4]}")
            suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=_dev)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, len(suffix_ids)), dtype=attention_mask.dtype, device=_dev),
            ], dim=1)
            with torch.inference_mode():
                outputs = model(
                    input_ids=suffix_tensor, attention_mask=attention_mask,
                    past_key_values=cache, logits_to_keep=1, use_cache=True)
                logits = outputs.logits[0, -1, :].float()
                cache = outputs.past_key_values
            _stats["decode_steps"] += 1
    _stats["completed_decisions"] += len(decisions)
    print(f"selector progress: {_stats['completed_decisions']}/{_total_decisions} decisions; "
          f"successful batches={_stats['forward_batches']} OOM splits={_stats['oom_splits']}",
          file=sys.stderr, flush=True)
    return choices

def score_choices(decisions):
    try:
        return _score_once(decisions)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(decisions) <= 1:
            raise
        _stats["oom_splits"] += 1
        print(f"selector OOM: splitting {len(decisions)} decisions into "
              f"{len(decisions[:len(decisions)//2])}+{len(decisions[len(decisions)//2:])}",
              file=sys.stderr, flush=True)
        torch.cuda.empty_cache()
        middle = len(decisions) // 2
        return {**score_choices(decisions[:middle]), **score_choices(decisions[middle:])}

thresholds = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
selected = []; threshold_selected = {f"{threshold:g}": [] for threshold in thresholds}
tot_dec = tot_chosen = tot_evidenced = 0
prepared = []
for branches, anchor_chunks, phone_chunks in zip(
        data["branch_transcripts"], data["anchor_chunks"], data["phone_evidence"]):
    active = [b for b in branches if b]
    evidence, graph = _evidence_by_slot(active, anchor_chunks, phone_chunks)
    prepared.append((graph, evidence))
_total_decisions = sum(len(build_decisions(graph, evidence_by_slot=evidence))
                       for graph, evidence in prepared)
print(f"selector plan: {_total_decisions} contested decisions across {len(prepared)} interviews",
      file=sys.stderr, flush=True)

for interview_number, (graph, evidence) in enumerate(prepared, 1):
    print(f"selector interview {interview_number}/{len(prepared)}: starting",
          file=sys.stderr, flush=True)
    _decision_scores.clear()
    text, nd, nc = select_graph_with_chooser(
        graph, score_choices, batch_size=24, evidence_by_slot=evidence)
    selected.append(text); tot_dec += nd; tot_chosen += nc; tot_evidenced += len(evidence)
    # Reassemble margin-gated diagnostic variants without another model forward pass. The zero
    # threshold is the canonical selector policy; references are only used later for reporting.
    for threshold in thresholds:
        gated = {slot: token for slot, (token, margin) in _decision_scores.items()
                 if margin >= threshold}
        gated_text = " ".join(selector_module.assemble(graph, gated)).strip()
        threshold_selected[f"{threshold:g}"].append(gated_text)
    print(f"selector interview {interview_number}/{len(prepared)}: complete; "
          "threshold variants reassembled from the existing graph",
          file=sys.stderr, flush=True)
finite_margins = [m for m in _stats["margins"] if math.isfinite(m)]
sorted_margins = sorted(finite_margins)
def _quantile(values, fraction):
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))] if values else None
stats = {"decisions": tot_dec, "scored": tot_chosen, "evidenced": tot_evidenced,
         "backbone_overrides": _stats["backbone_overrides"],
         "prefill_attempts": _stats["prefill_attempts"],
         "forward_batches": _stats["forward_batches"],
         "decode_steps": _stats["decode_steps"], "prompt_tokens": _stats["prompt_tokens"],
         "evidenced_backbone_overrides": _stats["evidenced_backbone_overrides"],
         "oom_splits": _stats["oom_splits"],
         "mean_logit_margin": (sum(finite_margins) / len(finite_margins)
                               if finite_margins else None),
         "p10_logit_margin": _quantile(sorted_margins, 0.10),
         "median_logit_margin": _quantile(sorted_margins, 0.50)}
print(f"selector restricted-scored {tot_chosen}/{tot_dec} contested slots; "
      f"overrode the Qwen backbone in {_stats['backbone_overrides']}; phone evidence attached to "
      f"{tot_evidenced}/{tot_dec}; forward batches={_stats['forward_batches']} "
      f"shared prompt tokens={_stats['prompt_tokens']} OOM splits={_stats['oom_splits']}",
      file=sys.stderr)
json.dump({"selected": selected, "threshold_selected": threshold_selected, "stats": stats},
          open(outp, "w"))
''')
        # Qwen's ~30 s timestamped text windows are the preferred retrieval anchor. If that branch
        # failed, fall back to another timestamped word branch; the phone windows remain immutable.
        _anchor_key = next((key for key, hyps in [
            ("A+B+Qwen3", hyp_Z), ("+Voxtral", hyp_C), ("+VoxtralVerbatim", hyp_VV),
            ("Whisper large-v3 quality", hyp_A3), ("A whisper", hyp_A),
            ("A+B", hyp_B)] if hyps and any(hyps) and key in WINDOW_PARTS), None)
        _anchors = WINDOW_PARTS.get(_anchor_key, [[] for _ in interviews])
        _selector_input = {
            # _wb is already Qwen-first. The worker preserves this order and anchors its graph to
            # branch 0, so abstentions and non-decisions retain Qwen rather than a majority vote.
            "branch_transcripts": [[h[k] for h in _wb] for k in range(len(interviews))],
            "anchor_chunks": _anchors,
            "phone_evidence": phone_evidence,
        }
        print(f"selector retrieval anchor: {_anchor_key or 'none'}; "
              f"phone windows={sum(len(rows) for rows in phone_evidence)}")
        json.dump(_selector_input, open(os.path.join(SEL_RUN, "in.json"), "w"))
        subprocess.run([SEL_VENV_PY, SEL_WORKER, SELECTOR_MODEL,
                        os.path.join(SEL_RUN, "in.json"), os.path.join(SEL_RUN, "out.json")], check=True)
        _selector_result = json.load(open(os.path.join(SEL_RUN, "out.json")))
        llm_selected = _selector_result["selected"]
        selector_stats = _selector_result["stats"]
        llm_selected_wer = wer_of(llm_selected)
        selector_threshold_wer = {
            threshold: round(wer_of(texts), 4)
            for threshold, texts in _selector_result["threshold_selected"].items()}
        canonical = llm_selected
        canonical_source = "constrained LLM selector"
        canonical_wer = llm_selected_wer
        print(f"[LLM SELECTED   ] canonical WER={llm_selected_wer:.3f}"
              f"   vs Qwen backbone={wer_of(backbone_hyps):.3f}"
              f"  vs baseline A={wer_of(hyp_A):.3f}"
              f"  vs realizable oracle={r_oracle:.3f}")
        gap = wer_of(hyp_A) - r_oracle
        print(f"selector captured {100*(wer_of(hyp_A)-llm_selected_wer)/max(gap,1e-9):.0f}% "
              f"of the baseline->realizable-oracle gap; delta vs Qwen backbone: "
              f"{llm_selected_wer-wer_of(backbone_hyps):+.3f}")
        print("selector stats:", json.dumps(selector_stats, sort_keys=True))
        print("diagnostic backbone-override margin curve (threshold -> WER):",
              json.dumps(selector_threshold_wer, sort_keys=True))
    except Exception as e:
        print(f"LLM selector failed; canonical transcript remains {backbone_name}:", repr(e)[:500])
    rec("+LLMSelector", time.time() - _sel_t0)
"""),
    md("""## 12. Timings + save summary
Persistent per-stage wall-times (every stage `⏱`-printed as it finished; here they're
collected into one table) and the WER summary (branch WERs, recall floor, realizable oracle,
Qwen backbone, and the canonical constrained-selector result when selection succeeds)."""),
    code(r"""
import json, sys, importlib.metadata as _metadata
from datetime import datetime, timezone
# --- timing table (slowest first) ---
print("=== wall-time by stage ===")
for name, dt in sorted(TIMINGS, key=lambda x: -x[1]):
    print(f"   {dt:7.1f}s  {name}")
print(f"   {sum(dt for _, dt in TIMINGS):7.1f}s  TOTAL (sum of tracked stages)")

res = {}
branch_metrics = {}
for name, h in [("A_whisper_turbo", hyp_A), ("A3_whisper_large_v3", hyp_A3),
                ("B_ctc", hyp_B), ("Z_qwen3", hyp_Z), ("C_voxtral", hyp_C),
                ("VV_voxtral_verbatim", hyp_VV), ("CW_crisperwhisper", hyp_CW),
                ("CWV_crisper_qwen_verbatize", hyp_CWV)]:
    if h and any(h):
        branch_metrics[name] = error_counts_of(h)
        branch_metrics[name]["interviews_with_text"] = sum(bool(text) for text in h)
        branch_metrics[name]["interviews_total"] = len(interviews)
        res[name] = branch_metrics[name]["wer"]

_package_names = ["classroom-asr", "torch", "transformers", "accelerate", "qwen-asr",
                  "faster-whisper", "mistral-common", "huggingface-hub", "rapidfuzz"]
_packages = {}
for _package in _package_names:
    try: _packages[_package] = _metadata.version(_package)
    except _metadata.PackageNotFoundError: pass

_model_ids = [FW_MODEL, FW_QUALITY_MODEL, CTC_MODEL, QWEN3ASR_MODEL, VOXTRAL_MODEL,
              f"nyralabs/CrisperWhisper2.0_{CRISPER_SIZE}", PHONE_MODEL,
              PHONETIC_XEUS_MODEL, SELECTOR_MODEL]
_resolved_hf_revisions = {}
try:
    from huggingface_hub import scan_cache_dir
    for _repo in scan_cache_dir().repos:
        if _repo.repo_id in _model_ids:
            _resolved_hf_revisions[_repo.repo_id] = sorted(
                revision.commit_hash for revision in _repo.revisions)
except Exception as _cache_error:
    _resolved_hf_revisions["_scan_error"] = type(_cache_error).__name__

_completed_epoch = _run_time.time()
run_fingerprint = {
    "source_git_ref": ASR_GIT_REF,
    "started_utc": datetime.fromtimestamp(RUN_STARTED_EPOCH, timezone.utc).isoformat(),
    "completed_utc": datetime.fromtimestamp(_completed_epoch, timezone.utc).isoformat(),
    "overall_wall_seconds": round(_completed_epoch - RUN_STARTED_EPOCH, 1),
    "tracked_stage_seconds": round(sum(dt for _, dt in TIMINGS), 1),
    "python": sys.version.split()[0],
    "packages": _packages,
    "hardware": {"gpu_count": torch.cuda.device_count(),
                 "gpu_names": [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())],
                 "cuda": torch.version.cuda},
    "models": {
        "whisper_turbo": {"id": FW_MODEL, "vad": WHISPER_VAD},
        "whisper_large_v3": {"id": FW_QUALITY_MODEL, "beam_size": 5,
                             "compute_type": "float16", "vad": WHISPER_VAD},
        "ctc": {"id": CTC_MODEL},
        "qwen3_asr": {"id": QWEN3ASR_MODEL, "language": "auto",
                      "window_seconds": QWEN_CHUNK_S,
                      "max_new_tokens": QWEN_MAX_NEW_TOKENS},
        "voxtral": {"id": VOXTRAL_MODEL, "language": "en",
                    "window_seconds": VOXTRAL_CHUNK_S},
        "crisper": {"id": f"nyralabs/CrisperWhisper2.0_{CRISPER_SIZE}",
                    "package_version": CRISPER_VERSION,
                    "qwen_verbatize": USE_CRISPER_VERBATIZE},
        "phone": {"wav2vec2": PHONE_MODEL, "phonetic_xeus": PHONETIC_XEUS_MODEL},
        "selector": {"enabled": USE_LLM_SELECTOR, "id": SELECTOR_MODEL,
                     "transformers": SELECTOR_TRANSFORMERS,
                     "accelerate": SELECTOR_ACCELERATE},
    },
    "resolved_hf_revisions": _resolved_hf_revisions,
}
summary = {"component": COMPONENT, "interviews": len(interviews), "minutes": round(total/60, 1),
           "scoring": "whole-recording; numbers+spelling folded; fillers kept",
           "branch_wer": res, "branch_metrics": branch_metrics,
           "baseline_wer": res.get("A_whisper_turbo"),
           # honest metrics: recall_floor = fraction of ref words no branch produced (a recall
           # LOWER BOUND, ignores insertions); realizable_oracle = best full-WER a selector over
           # the Qwen-anchored candidate graph could reach (a real transcript).
           "candidate_recall_floor": round(recall_floor(pool), 4),
           "realizable_oracle_wer": round(r_oracle, 4) if "r_oracle" in dir() else None,
           "primary_backbone": backbone_name if "backbone_name" in dir() else None,
           "qwen_backbone_wer": (round(wer_of(backbone_hyps), 4)
                                  if "backbone_hyps" in dir() else None),
           "canonical_source": canonical_source if "canonical_source" in dir() else None,
           "canonical_wer": round(canonical_wer, 4) if "canonical_wer" in dir() else None,
           "llm_selected_wer": (round(llm_selected_wer, 4)
                                if "llm_selected_wer" in dir()
                                and llm_selected_wer is not None else None),
           "llm_selector_stats": selector_stats if "selector_stats" in dir() else None,
           "llm_selector_margin_wer": (selector_threshold_wer
                                       if "selector_threshold_wer" in dir() else None),
           "branch_overlap_ablation": (branch_overlap_ablation
                                       if "branch_overlap_ablation" in dir() else None),
           "branch_pair_overlap": (branch_pair_overlap
                                   if "branch_pair_overlap" in dir() else None),
           "branch_subset_pareto": (branch_subset_pareto
                                     if "branch_subset_pareto" in dir() else None),
           "run_fingerprint": run_fingerprint,
           "timings_s": {name: round(dt, 1) for name, dt in TIMINGS}}
try:   # recall-floor breakdown from §11.5 (defined only if the pool had branches)
    summary["recall_floor_by_category"] = dict(bycat.most_common())
    summary["recall_floor_top"] = [[w, n] for w, n in floor.most_common(30)]
except NameError:
    pass
json.dump(summary, open("coraal_oracle_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
"""),
]

NB = {"cells": CELLS,
      "metadata": {"classroom_asr": {"kind": "payload", "schema": 1},
                   "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "| installs from github.com/" + GITHUB_REPO)
