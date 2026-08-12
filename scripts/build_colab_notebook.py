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
import torch
# Everything up front (mid-notebook installs don't reliably import on Kaggle). Two calls:
# loose deps first, then PIN transformers/accelerate LAST so qwen-asr's required
# versions win over anything crisperwhisper/others pull in (4.57.6 also fits Whisper/Voxtral).
%pip -q install "mistral-common[audio]" phonemizer faster-whisper qwen-asr "crisperwhisper[ct2]" soundfile rapidfuzz
%pip -q install "transformers==4.57.6" "accelerate==1.12.0" "git+https://github.com/{GITHUB_REPO}.git"
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

# Speed notes: the LLM branches auto-pick fp16 on T4 (Turing has no bf16 tensor cores —
# that was most of the slowness). Voxtral is the heaviest; each Voxtral toggle is a full
# 9.5 GB load + a whole-hour pass, so set USE_VOXTRAL=False to run only the verbatim one.
# CRISPER_SIZE="turbo" is faster than "large" if you want to trade a little quality.

# Window (seconds) for the audio-LLM branches (Qwen3, Voxtral). These emit a BOUNDED
# number of output tokens per call, so an over-long window truncates the transcript
# (600 s gave Qwen3 WER 0.87 — near-total deletion). Keep it near the utterance scale,
# like Whisper's internal 30 s window; this is capped by output budget, not GPU memory.
LLM_CHUNK_S    = 30

BASE = f"http://lingtools.uoregon.edu/coraal/{COMPONENT}/{VERSION}"
COMP = COMPONENT.upper()
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
    hyp_Z = run_branch("A+B+Qwen3", lambda dev: Qwen3ASR(
        QWEN3ASR_MODEL, language="English", device=dev),
        get_text=lambda m, a: m.transcribe_full(a, chunk_s=LLM_CHUNK_S))
    add_branch("A+B+Qwen3", hyp_Z)
"""),
    md("## 9. Branch C — Voxtral Mini 3B (audio-LLM)"),
    code(r"""
hyp_C = None
if USE_VOXTRAL:
    from classroom_asr.backends.voxtral_asr import VoxtralASR
    hyp_C = run_branch("+Voxtral", lambda dev: VoxtralASR(VOXTRAL_MODEL, language="en", device=dev),
                       get_text=lambda m, a: m.transcribe_full(a, chunk_s=LLM_CHUNK_S))
    add_branch("+Voxtral", hyp_C)
"""),
    md("""## 9a. Branch CW — CrisperWhisper 2.0 (**verbatim** Whisper)
The verbatim lever: a Whisper fine-tune that *keeps* the `um`/`uh`/false starts the
clean branches delete — exactly the deletions that dominate the residual WER on a
verbatim reference. Its CT2 runtime does its own long-form windowing, so it's as fast
as the Whisper baseline. Watch whether it drops `oracle(pool)` more than the clean
LLM branches did."""),
    code(r"""
hyp_CW = None
if USE_CRISPER:
    from classroom_asr.backends.crisperwhisper_asr import CrisperWhisperV2
    hyp_CW = run_branch("+CrisperWhisper",
                        lambda dev: CrisperWhisperV2(CRISPER_SIZE, device=dev))
    add_branch("+CrisperWhisper", hyp_CW)
"""),
    md("""## 9b. Branch VV — Voxtral, **verbatim-prompted**
The same Voxtral checkpoint, but in instruct mode told to transcribe verbatim (keep
every filler, false start, repetition). A second, independent verbatim source — an
audio-LLM's take on the same job CrisperWhisper does acoustically."""),
    code(r"""
hyp_VV = None
if USE_VOXTRAL_VERBATIM:
    from classroom_asr.backends.voxtral_asr import VoxtralASR
    hyp_VV = run_branch("+VoxtralVerbatim",
                        lambda dev: VoxtralASR(VOXTRAL_MODEL, language="en", device=dev,
                                               mode="verbatim"),
                        get_text=lambda m, a: m.transcribe_full(a, chunk_s=LLM_CHUNK_S))
    add_branch("+VoxtralVerbatim", hyp_VV)
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
