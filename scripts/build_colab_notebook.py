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
import torch
# TF32 matmul: free speedup on Ampere+ (Colab A100/L4), no-op on T4/Turing; inference-safe.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Everything up front (mid-notebook installs don't reliably import on Kaggle). Two calls:
# loose deps first, then PIN transformers/accelerate LAST so qwen-asr's required
# versions win over anything crisperwhisper/others pull in (4.57.6 also fits Whisper/Voxtral).
%pip -q install "mistral-common[audio]" phonemizer faster-whisper qwen-asr soundfile rapidfuzz hf_transfer
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
now, so they finish *while* the A→Qwen branches compute. Pure overlap — no GPU used here."""),
    code(r"""
import os, sys, subprocess, threading
# (a) CrisperWhisper CT2 venv — built in the background; §9a just joins this thread.
CW_WORK = os.path.abspath("cw_iso"); os.makedirs(CW_WORK, exist_ok=True)
CW_VENV = os.path.join(CW_WORK, "venv"); CW_VENV_PY = os.path.join(CW_VENV, "bin", "python")
_cw = {"thread": None, "err": None}
def _cw_prewarm():
    try:
        if not os.path.exists(CW_VENV_PY):
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", CW_VENV], check=True)
            subprocess.run([os.path.join(CW_VENV, "bin", "pip"), "-q", "install",
                            "crisperwhisper[ct2]"], check=True)
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
"""),
    md("""## 5. Whole-recording runner + scoring (rapidfuzz)
`whole_rec` transcribes each interview on the GPUs. `wer_of` scores a branch over the
full transcripts; `oracle_wer` is the candidate-oracle at the word level — the fraction
of reference words that **no** branch recovers (a lower bound the pool can't beat)."""),
    code(r"""
import threading, gc
from tqdm.auto import tqdm
from rapidfuzz.distance import Levenshtein
from classroom_asr.normalize import Normalizer
SCORE = Normalizer(fold_numbers=True, fold_spelling=True)   # keep fillers; fold formatting

def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def whole_rec(make_model, get_text, desc):
    out = [""] * len(interviews)
    models = [make_model("cpu" if g is None else f"cuda:{g}") for g in GPUS]
    shards = [list(range(len(interviews)))[i::len(GPUS)] for i in range(len(GPUS))]
    def worker(model, sh, pos):
        for k in tqdm(sh, desc=f"{desc}:{pos}", position=pos, leave=False):
            try: out[k] = get_text(model, load_16k(interviews[k][0]))
            except Exception as e: print(desc, "failed:", repr(e)[:120]); out[k] = ""
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(GPUS))]
    for t in ts: t.start()
    for t in ts: t.join()
    for m in models:
        if hasattr(m, "unload"):
            try: m.unload()
            except Exception as e: print(desc, "unload failed:", repr(e)[:100])
    models = None; del models; _free()   # release GPU memory before the next branch loads
    if torch.cuda.is_available():        # empty each device's cache, not just current
        for g in GPUS:
            if g is not None:
                with torch.cuda.device(g): torch.cuda.empty_cache()
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

# window_pass with its own model load/unload + per-branch error guard.
def run_windows(desc, make_model, *, chunk_s, batch_size):
    try:
        models = [make_model("cpu" if g is None else f"cuda:{g}") for g in GPUS]
    except Exception as e:
        print(f"[{desc:14s}] load FAILED: {repr(e)[:160]}"); _free(); return [""] * len(interviews)
    try:
        return window_pass(models, desc, chunk_s=chunk_s, batch_size=batch_size)
    except Exception as e:
        print(f"[{desc:14s}] FAILED: {repr(e)[:160]}"); return [""] * len(interviews)
    finally:
        _free_models(models)

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
hyp_A = run_branch("A whisper", lambda dev: FasterWhisperASR(
    FW_MODEL, language="en", device=dev, vad_filter=WHISPER_VAD))
pool = []
add_branch("A", hyp_A)
print("^ baseline WER = branch A")
"""),
    md("## 7. Branch B — wav2vec2 CTC"),
    code(r"""
hyp_B = None
if USE_CTC:
    from classroom_asr.backends.wav2vec2_ctc import Wav2Vec2CTC
    hyp_B = run_branch("A+B", lambda dev: Wav2Vec2CTC(CTC_MODEL, device=dev))
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
                        chunk_s=LLM_CHUNK_S, batch_size=16)
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
    vmodels = [VoxtralASR(VOXTRAL_MODEL, language="en",
                          device=("cpu" if g is None else f"cuda:{g}")) for g in GPUS]
    def _vox_pass(mode, tag):                       # windows balanced across GPUs, one mode
        for m in vmodels: m.mode = mode
        return window_pass(vmodels, tag, chunk_s=LLM_CHUNK_S, batch_size=6)
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
the main kernel never imports the fork."""),
    code(r"""
hyp_CW = None
if USE_CRISPER:
    import os, sys, json, subprocess
    WORKER = os.path.join(CW_WORK, "cw_worker.py")
    try:
        if _cw["thread"] is not None:
            _cw["thread"].join()                 # wait for the background venv build (§2a)
        if _cw["err"] is not None:
            raise _cw["err"]
        with open(WORKER, "w") as f:
            f.write('''
import sys, json
from crisperwhisper import CrisperWhisperModel
size, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
paths = json.load(open(inp))
try:
    m = CrisperWhisperModel(size, backend="ct2")          # fast forked CT2
except Exception as e:
    print("ct2 unavailable, transformers backend:", repr(e)[:120], file=sys.stderr)
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
        json.dump(paths, open(os.path.join(CW_WORK, "in.json"), "w"))
        subprocess.run([CW_VENV_PY, WORKER, CRISPER_SIZE,
                        os.path.join(CW_WORK, "in.json"), os.path.join(CW_WORK, "out.json")], check=True)
        res = json.load(open(os.path.join(CW_WORK, "out.json")))
        from classroom_asr.backends.crisperwhisper_asr import CrisperWhisperV2
        hyp_CW = [CrisperWhisperV2._clean(res.get(p, "")) for p in paths]   # [um]->um, drop [laughter]
    except Exception as e:
        print("CrisperWhisper isolation failed:", repr(e)[:200]); hyp_CW = [""] * len(interviews)
    add_branch("+CrisperWhisper", hyp_CW)
"""),
    md("""## 10. Phone branch — realized IPA (the pronunciation path)
Not a word transcript: the phone branch's product is *pronunciation*. Scoring it needs a
phonetic reference (PER/IPA-CER, §18.1), which CORAAL doesn't provide — so we show the
realized IPA rather than force it through a naive P2G into a misleading word WER. It's a
first-class part of the design (OOV/nonce recovery), just not exercised by this dataset."""),
    code(r"""
if USE_PHONE:
    from classroom_asr.backends.wav2vec2_phone import Wav2Vec2Phone
    ipa = whole_rec(lambda dev: Wav2Vec2Phone(PHONE_MODEL, device=dev),
                    lambda m, a: m.transcribe_full(a), "phone")
    print("realized IPA (first 300 chars of interview 0):")
    print(" ", (ipa[0] or "")[:300])
    print("\nreference words (for contrast):", refs[0][:200])
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
    md("## 12. Save summary"),
    code(r"""
import json
res = {}
for name, h in [("A_whisper", hyp_A), ("B_ctc", hyp_B), ("Z_qwen3", hyp_Z), ("C_voxtral", hyp_C)]:
    if h and any(h): res[name] = round(wer_of(h), 4)
summary = {"component": COMPONENT, "interviews": len(interviews), "minutes": round(total/60, 1),
           "scoring": "whole-recording; numbers+spelling folded; fillers kept",
           "branch_wer": res, "oracle_wer": round(oracle_wer(pool), 4),
           "baseline_wer": res.get("A_whisper")}
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
