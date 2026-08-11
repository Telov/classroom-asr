"""Generate the CORAAL candidate-oracle notebook (Colab or Kaggle).

Installs the classroom_asr package from GitHub (`pip install git+...`) so a fresh
notebook needs nothing uploaded. Uses all available GPUs (e.g. Kaggle's 2x T4) by
sharding segments across devices in threads.

Run:  python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

GITHUB_REPO = "Telov/classroom-asr"   # pip install git+https://github.com/<this>.git
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "CORAAL_candidate_oracle.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(r"""
# CORAAL candidate-oracle WER — real audio, *our* stack (Colab **or** Kaggle)

Runs the candidate-oracle WER gate (design §18.2) on ~1 h of professionally
transcribed, verbatim conversation (CORAAL), with **real ASR models** behind our
`AcousticModel` interface. Branches (each a *different architecture*, so their
errors are complementary — §8):

* **A** Whisper large-v3-turbo via **faster-whisper / CTranslate2 int8** — 1-best,
  several times faster than HF Whisper
* **B** wav2vec2 CTC (acoustic CTC, no LM)
* **Z** Qwen3-ASR-1.7B — the design's real backbone (multilingual LLM-ASR), **batched**
* **C** Voxtral Mini 3B (audio-LLM) — **batched**, capped to a time subset
* **phone** wav2vec2 phoneme CTC → IPA lattice + naive P2G (the phonetic path)

**Kaggle:** Settings → Accelerator → **GPU T4 x2**, and turn **Internet ON**
(needed for pip + CORAAL + model downloads). This notebook uses **both** T4s.

**Speed:** every branch is batched/sharded across both GPUs. Whisper runs via
faster-whisper (CTranslate2 int8), Qwen3-ASR batches natively, and Voxtral is batched
and capped to `VOXTRAL_MINUTES` of audio. Toggle `USE_FASTER_WHISPER`, `USE_VOXTRAL`,
`VOXTRAL_MINUTES`, `VOX_BATCH` to trade coverage for time.

**We read the *gap* between baseline and oracle, not absolute WER** — CORAAL is
spontaneous English, so WER is high and deletion-heavy (Whisper trims fillers
CORAAL keeps). The gap is what generalizes to the design decision.
"""),
    md("## 1. Install (from GitHub — nothing to upload)"),
    code(f"""
import torch
# Install EVERYTHING up front: mid-notebook pip installs don't reliably become
# importable in Kaggle's running kernel (that's why Voxtral/phone were skipping).
# Pin transformers to what qwen-asr needs (==4.57.6); it also satisfies Whisper &
# Voxtral (>=4.54), so all branches share one compatible stack.
!apt-get -qq install -y espeak-ng > /dev/null 2>&1          # phonemizer runtime (phone branch)
!pip -q install "transformers==4.57.6" "accelerate==1.12.0" soundfile
!pip -q install "mistral-common[audio]" phonemizer faster-whisper qwen-asr
!pip -q install "git+https://github.com/{GITHUB_REPO}.git"
import classroom_asr

# Warnings are fixed at the source in the backends (generation flags, attention
# mask, max_length, dtype, tokenizer cleanup) rather than blanket-silenced, so a
# genuinely new warning would still surface. Optional: set HF_TOKEN (Kaggle:
# Add-ons -> Secrets) to raise HF download limits.
import os
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login; login(os.environ["HF_TOKEN"])

GPUS = list(range(torch.cuda.device_count())) or [None]
print("classroom_asr", classroom_asr.__version__, "| GPUs:", torch.cuda.device_count(),
      "|", [torch.cuda.get_device_name(g) for g in range(torch.cuda.device_count())])
"""),
    md("## 2. Parameters"),
    code(r"""
# --- CORAAL component (both verified live) -------------------------------
COMPONENT, VERSION, AUDIO_PART = "les", "2021.07", "part01"   # or "prv","2018.10.06"
TARGET_MINUTES = 60
INCLUDE_INTERVIEWER = True

# --- branches (toggle to trade accuracy vs GPU time) ---------------------
USE_FASTER_WHISPER = True  # CTranslate2 int8 — ~5-8x faster than HF Whisper
FW_MODEL      = "deepdml/faster-whisper-large-v3-turbo-ct2"
PRIMARY_MODEL = "openai/whisper-large-v3-turbo"   # HF fallback if USE_FASTER_WHISPER=False
N_BEST_A      = 1          # 1 = fast (Whisper beams add ~no oracle diversity)
BATCH_SIZE    = 16

USE_CTC       = True
CTC_MODEL     = "facebook/wav2vec2-large-960h-lv60-self"

USE_QWEN3ASR  = True       # the design's real backbone (small, fast)
QWEN3ASR_MODEL= "Qwen/Qwen3-ASR-1.7B"

USE_VOXTRAL   = True       # heavy audio-LLM; batched + capped to a time subset
VOXTRAL_MODEL = "mistralai/Voxtral-Mini-3B-2507"
VOXTRAL_MINUTES = 20       # time cap (raise for full coverage)
VOX_BATCH     = 4          # Voxtral batch size (3B LLM; smaller to fit VRAM)

USE_PHONE     = True
PHONE_MODEL   = "facebook/wav2vec2-lv-60-espeak-cv-ft"

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
    with tarfile.open(tarball) as t: t.extractall(dest, filter="data")  # py3.12+ safe extract
print("transcripts:", len(list(Path('coraal/txt').rglob('*.txt'))),
      "| wavs:", len(list(Path('coraal/audio').rglob('*.wav'))))
"""),
    md("## 4. Select ~1h of segments + slice audio (16 kHz mono)"),
    code(r"""
import torchaudio, numpy as np
from pathlib import Path
from classroom_asr.data.coraal import parse_transcript, select_segments, iter_transcripts

wavs = {p.stem: p for p in Path("coraal/audio").rglob("*.wav")}
budget = TARGET_MINUTES*60.0; plan=[]; total=0.0
for txt in iter_transcripts("coraal/txt"):
    if txt.stem not in wavs: continue
    keep = select_segments(parse_transcript(txt), max_seconds=budget-total,
                           include_interviewer=INCLUDE_INTERVIEWER)
    if keep: plan.append((wavs[txt.stem], keep)); total += sum(s.duration for s in keep)
    if total >= budget: break

_full={}
def load_16k(path):
    if path in _full: return _full[path]
    w, sr = torchaudio.load(str(path))
    if w.shape[0] > 1: w = w.mean(0, keepdim=True)
    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
    a = w.squeeze(0).numpy().astype("float32"); _full[path]=a; return a

flat = [(w,s) for w,segs in plan for s in segs]
flat_audio = [load_16k(w)[max(0,int(s.start*16000)):int(s.end*16000)] for (w,s) in flat]
refs = [s.text for (w,s) in flat]
_full.clear()
print(f"{len(plan)} interviews | {len(flat)} segments | {total/60:.1f} min")
"""),
    md("""## 5. Multi-GPU sharded runners
Split segments across every GPU (2 on Kaggle) and run in threads. Falls back to a
single device automatically."""),
    code(r"""
import threading
from tqdm.auto import tqdm
from classroom_asr.datamodel import Span
from classroom_asr.timeline import Interval
from classroom_asr.types import CandidateSource, Role, AudioSource
from classroom_asr.candidates import mbr_consensus, span_candidate_texts
from classroom_asr.metrics import candidate_oracle_wer, corpus_wer, score
from classroom_asr.normalize import Normalizer

# Scoring normalizer: fold numbers/ordinals/spelling (formatting noise) but KEEP
# fillers/function words (their deletion is a real metric). See §18/§29.
SCORE = Normalizer(fold_numbers=True, fold_spelling=True)

_valid = [k for k,a in enumerate(flat_audio) if a is not None and len(a) > 400]

def _dev(g): return "cpu" if g is None else f"cuda:{g}"

def time_limited_subset(limit_seconds):
    acc, out = 0.0, []
    for k in _valid:
        if acc > limit_seconds: break
        out.append(k); acc += len(flat_audio[k]) / 16000
    return out

def sharded_batched(make_model, infer_batch, desc, subset=None, bs=None):
    idxs = subset if subset is not None else _valid
    out = [[] for _ in flat_audio]
    models = [make_model(_dev(g)) for g in GPUS]
    shards = [idxs[i::len(GPUS)] for i in range(len(GPUS))]
    def worker(model, sh, pos):
        bs_, i = (bs or BATCH_SIZE), 0
        pbar = tqdm(total=len(sh), desc=f"{desc}:{pos}", position=pos, leave=False)
        while i < len(sh):
            bk = sh[i:i+bs_]
            try: res = infer_batch(model, [flat_audio[k] for k in bk])
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and bs_ > 1:
                    torch.cuda.empty_cache(); bs_ = max(1, bs_//2); continue
                raise
            for k,r in zip(bk,res): out[k]=r
            i += len(bk); pbar.update(len(bk))
        pbar.close()
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(GPUS))]
    for t in ts: t.start()
    for t in ts: t.join()
    for m in models:
        if hasattr(m,'unload'): m.unload()
    return out

def uncertain_indices(bA, bB):
    # §12.5 expansion policy: a span is "uncertain" if the cheap branches
    # (Whisper vs CTC) disagree under the scoring normalizer, or CTC is empty.
    # Expensive branches only need to run on these.
    out = []
    for k in _valid:
        a = bA[k][0].text if bA and bA[k] else ""
        b = bB[k][0].text if bB and bB[k] else ""
        if (not b) or (SCORE.tokens(a) != SCORE.tokens(b)):
            out.append(k)
    return out

def sharded_seq(make_model, infer_one, desc, limit_seconds=None, only=None):
    idxs = list(only) if only is not None else list(_valid)
    if limit_seconds:
        acc, sub = 0.0, []
        for k in idxs:
            if acc > limit_seconds: break
            sub.append(k); acc += len(flat_audio[k])/16000
        idxs = sub
    out = [[] for _ in flat_audio]
    models = [make_model(_dev(g)) for g in GPUS]
    shards = [idxs[i::len(GPUS)] for i in range(len(GPUS))]
    def worker(model, sh, pos):
        for k in tqdm(sh, desc=f"{desc}:{pos}", position=pos, leave=False):
            try: out[k] = infer_one(model, flat_audio[k])
            except Exception as e: pass
    ts = [threading.Thread(target=worker, args=(models[i], shards[i], i)) for i in range(len(GPUS))]
    for t in ts: t.start()
    for t in ts: t.join()
    for m in models:
        if hasattr(m,'unload'): m.unload()
    return out

def build_spans(bA, bB=None, extras=(), phones=None, p2g=None):
    spans, rf = [], []
    for k,(w,s) in enumerate(flat):
        qA = bA[k]; gB = (bB[k] if bB else [])
        xt = [c for br in extras if br for c in (br[k] or [])]
        ph = (phones[k] if phones else []); px = (p2g[k] if p2g else [])
        if not (qA or gB or xt): continue
        sp = Span(span_id=len(spans), speaker=Role.STUDENT, audio_source=AudioSource.STUDENT_RAW,
                  interval=Interval(s.start,s.end), qwen=qA, gigaam=gB, extra=xt, phones=ph, p2g=px)
        m = mbr_consensus([*qA,*gB,*xt])
        if m: sp.mbr=[m]
        spans.append(sp); rf.append(refs[k])
    return spans, rf

def report(tag, spans, rf):
    base = [sp.qwen[0].text if sp.qwen else (sp.gigaam[0].text if sp.gigaam else "") for sp in spans]
    o = candidate_oracle_wer(rf, [span_candidate_texts(sp) for sp in spans],
                             baseline_choice=base, norm=SCORE)
    print(f"[{tag:14s}] segs={len(spans):4d}  baseline={corpus_wer(zip(rf,base), norm=SCORE).wer:.3f}  "
          f"oracle={o.oracle.wer:.3f}  headroom={o.headroom:+.3f}")
    return o
"""),
    md("""## 6. Branch A — Whisper (1-best, multi-GPU) → oracle printed here
Uses **faster-whisper** (CTranslate2 int8) by default — several times faster than HF
Whisper because it avoids the slow 30 s-padded encoder passes on tiny clips."""),
    code(r"""
if USE_FASTER_WHISPER:
    from classroom_asr.backends.faster_whisper_asr import FasterWhisperASR
    branch_A = sharded_seq(
        lambda dev: FasterWhisperASR(FW_MODEL, id_prefix="q", source=CandidateSource.QWEN,
                                     language="en", device=dev),
        lambda m, a: m.nbest(a), "fwhisperA")
else:
    from classroom_asr.backends.whisper_asr import WhisperASR
    branch_A = sharded_batched(
        lambda dev: WhisperASR(PRIMARY_MODEL, id_prefix="q", source=CandidateSource.QWEN,
                               language="en", device=dev),
        lambda m, wavs: m.nbest_batch(wavs, n_best=N_BEST_A), "whisperA")
spans, rf = build_spans(branch_A)
oracle_all = report("A", spans, rf)
# raw (no folding) baseline for contrast — shows how much WER was pure formatting
_raw = corpus_wer(zip(rf, [sp.qwen[0].text if sp.qwen else "" for sp in spans])).wer
print(f"   (branch-A baseline without number/spelling folding: {_raw:.3f})")
"""),
    md("## 7. Branch B — wav2vec2 CTC (different architecture, cheap)"),
    code(r"""
branch_B = None
if USE_CTC:
  try:
    from classroom_asr.backends.wav2vec2_ctc import Wav2Vec2CTC
    branch_B = sharded_batched(
        lambda dev: Wav2Vec2CTC(CTC_MODEL, id_prefix="c", source=CandidateSource.GIGAAM, device=dev),
        lambda m, wavs: m.nbest_batch(wavs, n_best=1), "ctcB")
    spans, rf = build_spans(branch_A, branch_B)
    oracle_all = report("A+B", spans, rf)
  except Exception as e:
    branch_B = None; print("branch B (CTC) skipped:", repr(e)[:200])
"""),
    md("## 8. Branch Z — Qwen3-ASR-1.7B (the design's real backbone)"),
    code(r"""
branch_Z = None
if USE_QWEN3ASR:
  try:
    from classroom_asr.backends.qwen3_asr import Qwen3ASR
    # qwen-asr batches natively -> use the batched sharded runner (full coverage, fast)
    branch_Z = sharded_batched(
        lambda dev: Qwen3ASR(QWEN3ASR_MODEL, id_prefix="z", source=CandidateSource.QWEN,
                             language="English", device=dev, max_inference_batch_size=BATCH_SIZE),
        lambda m, wavs: m.nbest_batch(wavs, n_best=1), "qwen3asr")
    spans, rf = build_spans(branch_A, branch_B, extras=[branch_Z])
    oracle_all = report("A+B+Qwen3", spans, rf)
  except Exception as e:
    branch_Z = None; print("branch Z (Qwen3-ASR) skipped:", repr(e)[:200])
"""),
    md("## 9. Branch C — Voxtral Mini 3B (audio-LLM, subset)"),
    code(r"""
branch_C = None
if USE_VOXTRAL:
  try:
    from classroom_asr.backends.voxtral_asr import VoxtralASR
    vox_idx = time_limited_subset(VOXTRAL_MINUTES * 60)   # ungated; just a time cap
    print(f"Voxtral on {len(vox_idx)}/{len(_valid)} spans (first {VOXTRAL_MINUTES} min), batched")
    branch_C = sharded_batched(
        lambda dev: VoxtralASR(VOXTRAL_MODEL, id_prefix="v", source=CandidateSource.QWEN,
                               language="en", device=dev),
        lambda m, wavs: m.nbest_batch(wavs, n_best=1), "voxtral",
        subset=vox_idx, bs=VOX_BATCH)
    spans, rf = build_spans(branch_A, branch_B, extras=[branch_Z, branch_C])
    oracle_all = report("+Voxtral", spans, rf)
  except Exception as e:
    branch_C = None; print("branch C (Voxtral) skipped:", repr(e)[:200])
"""),
    md("## 10. Phone / P2G path — wav2vec2 phoneme CTC (design's PhoneticXEUS slot)"),
    code(r"""
phones = p2g = None
if USE_PHONE:
  try:
    from classroom_asr.backends.wav2vec2_phone import Wav2Vec2Phone
    from classroom_asr.pipeline.stubs import StubP2G
    phones = sharded_batched(
        lambda dev: Wav2Vec2Phone(PHONE_MODEL, device=dev),
        lambda m, wavs: m.recognize_batch(wavs, top_k=1), "phone")
    _p2g = StubP2G()
    p2g = [_p2g.convert(ph, n_best=2) if ph else [] for ph in phones]
    spans, rf = build_spans(branch_A, branch_B, extras=[branch_Z, branch_C], phones=phones, p2g=p2g)
    got = sum(1 for ph in phones if ph)
    print("phone paths on", got, "segments; e.g. IPA:",
          next((ph[0].ipa for ph in phones if ph), "-")[:40])
    oracle_all = report("ALL+phone", spans, rf)
  except Exception as e:
    phones = p2g = None; print("phone branch skipped:", repr(e)[:200])
"""),
    md("""## 11. **What specific mistakes were made** — error analysis
Breaks the 1-best (baseline) and oracle down into substitutions / deletions /
insertions, the most-dropped words, common confusions, and worst utterances."""),
    code(r"""
from classroom_asr.erroranalysis import error_report, worst_utterances

baseline = [sp.qwen[0].text if sp.qwen else "" for sp in spans]
chosen   = list(oracle_all.chosen)

print("================ BASELINE (branch-A 1-best) mistakes ================")
print(error_report(zip(rf, baseline), norm=SCORE).format(top=12))
print("\n================ ORACLE (best candidate per span) residual ========")
print(error_report(zip(rf, chosen), norm=SCORE).format(top=12))

print("\n================ Worst baseline utterances ========================")
for ref, hyp, w in worst_utterances(zip(rf, baseline), n=12, norm=SCORE):
    print(f"WER {w:.2f}\n  REF: {ref}\n  HYP: {hyp}")
"""),
    md("## 12. Where the candidate pool rescued the 1-best (which branch helped)"),
    code(r"""
shown = 0
for sp, ref, ch in zip(spans, rf, oracle_all.chosen):
    b = sp.qwen[0].text if sp.qwen else ""
    if score(ref, ch).wer < score(ref, b).wer:
        # tag which branch produced the winning candidate
        src = "?"
        for c in sp.selectable_candidates():
            if c.text == ch: src = c.source.value + "/" + c.id; break
        print(f"[{src}]\n  REF   : {ref}\n  1-best: {b}\n  oracle: {ch}")
        print("  pool  :", span_candidate_texts(sp)[:8]); print("-"*70)
        shown += 1
    if shown >= 15: break
print(f"(showing {shown} rescued spans)")
"""),
    md("## 13. Save summary"),
    code(r"""
import json
branches = ["A:"+PRIMARY_MODEL]
if USE_CTC: branches.append("B:"+CTC_MODEL)
if USE_QWEN3ASR: branches.append("Z:"+QWEN3ASR_MODEL)
if USE_VOXTRAL: branches.append(f"C:{VOXTRAL_MODEL}({VOXTRAL_MINUTES}min)")
if USE_PHONE: branches.append("phone:"+PHONE_MODEL)
base_res = corpus_wer(zip(rf, baseline), norm=SCORE)
summary = {"component": COMPONENT, "minutes": round(total/60,1), "segments": len(spans),
           "branches": branches, "scoring": "numbers+ordinals+spelling folded; fillers kept",
           "baseline_wer": base_res.wer,
           "oracle_wer": oracle_all.oracle.wer, "headroom": oracle_all.headroom,
           "baseline_del": base_res.deletions, "baseline_sub": base_res.substitutions,
           "baseline_ins": base_res.insertions}
json.dump(summary, open("coraal_oracle_summary.json","w"), indent=2)
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
