from classroom_asr.lexicon import Lexicon, combined_query
from classroom_asr.phonetics import phonetic_similarity
from classroom_asr.types import Role


def test_phonetic_similarity_identity_and_near_class():
    assert phonetic_similarity("θriː", "θriː") == 1.0
    # /θ/ ~ /s/ is a cheap substitution, so "three" vs "sri" stays high
    assert phonetic_similarity("θriː", "sriː") > 0.7
    assert phonetic_similarity("aboba", "xyz") < 0.5


def test_observe_and_query():
    lex = Lexicon()
    lex.observe("aboba", "ɐbobə", role=Role.TEACHER, time=41.0,
                canonical_spelling="aboba", exact=True, nonce=True)
    matches = lex.query("ɐbobə", top_k=3, min_similarity=0.5)
    assert matches
    assert matches[0].term == "aboba"
    assert matches[0].similarity >= 0.85
    entry = lex.get("aboba")
    assert entry.exact and entry.nonce


def test_exact_upgrade():
    lex = Lexicon()
    lex.observe("chel", "t͡ʃel", exact=False)
    assert not lex.get("chel").exact
    lex.add_term("chel", exact=True)
    assert lex.get("chel").exact


def test_combined_query_dedup_session_wins():
    session = Lexicon()
    persistent = Lexicon(persistent=True)
    session.observe("aboba", "ɐbobə")
    persistent.observe("aboba", "aboba")
    out = combined_query(session, persistent, "ɐbobə", top_k=5, min_similarity=0.3)
    terms = [m.term for m in out]
    assert terms.count("aboba") == 1        # deduped


def test_roundtrip(tmp_path):
    lex = Lexicon()
    lex.observe("aboba", "ɐbobə", nonce=True, exact=True, canonical_spelling="aboba")
    p = tmp_path / "lex.json"
    lex.save(p)
    back = Lexicon.load(p)
    assert back.get("aboba").nonce
    assert back.query("ɐbobə")[0].term == "aboba"
