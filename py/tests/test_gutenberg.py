"""Project Gutenberg cleaner: synthetic fixtures (authored for this repo)."""
from pathlib import Path

from intgrad.corpus import gutenberg as g

BODY = "\n\n".join(
    f"Chapter {i}. The lighthouse keeper walked along the shore and counted "
    f"the waves, as he had done every evening for {i + 10} years, and the "
    "gulls answered him from the rocks below the old stone wall."
    for i in range(1, 80)
).encode()


def wrap(head: bytes, body: bytes, tail: bytes) -> bytes:
    return head + b"\n\n" + body + b"\n\n" + tail


def test_modern_markers_wrapped_title_and_credits():
    head = (b"The Project Gutenberg EBook of The Keeper, by A. Writer\n\n"
            b"This eBook is for the use of anyone anywhere at no cost.\n\n"
            b"*** START OF THIS PROJECT GUTENBERG EBOOK THE KEEPER OF THE\n"
            b"NORTHERN LIGHT ***\n\n\n"
            b"Produced by Jane Doe and the Online Distributed\n"
            b"Proofreading Team at http://www.pgdp.net\n\n"
            b"THE KEEPER\n\nA NOVEL")
    tail = (b"THE END\n\n\n\nEnd of Project Gutenberg's The Keeper, by A. Writer\n\n"
            b"*** END OF THIS PROJECT GUTENBERG EBOOK THE KEEPER ***\n\n"
            b"Updated editions will replace the previous one.")
    out, st = g.clean(wrap(head, BODY, tail))
    assert out.startswith(b"THE KEEPER\n\nA NOVEL")       # title page kept
    assert out.endswith(b"THE END")
    assert b"NORTHERN LIGHT" not in out                    # wrapped marker line
    assert b"Proofreading" not in out and b"Gutenberg" not in out
    assert BODY in out
    assert not st.guarded


def test_old_small_print_header_without_start_marker():
    head = (b"The Project Gutenberg Etext of The Keeper\n\n"
            b"Copyright laws are changing all over the world, be sure to check\n"
            b"the laws for your country before redistributing these files!!!\n\n"
            b"Please do not remove this.\n\n"
            b"**The Legal Small Print**\n\n"
            b"*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*\n\n"
            b"THE KEEPER")
    out, _ = g.clean(wrap(head, BODY, b"End of the Project Gutenberg Etext of The Keeper"))
    assert out.startswith(b"THE KEEPER")
    assert out.endswith(BODY[-60:])


def test_transcriber_notes_head_tail_and_inline():
    head = (b"[Transcriber's Note:\n\nItalics are shown as _underscores_.\n\n"
            b"Obvious printer errors have been repaired.]\n\nTHE KEEPER")
    body = BODY.replace(b"Chapter 40.", b"Chapter 40 [Transcriber's note: sic].", 1)
    tail = (b"FINIS\n\nTranscriber's Notes\n\n"
            b"Page 12, \"keepr\" changed to \"keeper\".\n\n"
            b"Page 99, missing period added.")
    out, st = g.clean(wrap(head, body, tail))
    assert out.startswith(b"THE KEEPER")
    assert out.endswith(b"FINIS")
    assert b"Transcriber" not in out and b"underscores" not in out
    assert b"Chapter 40 ." in out or b"Chapter 40." in out
    assert st.inline == 1


def test_book_prose_mentioning_production_is_kept():
    # "produced by" inside prose, "transcribers" as a word, a mid-book line
    # starting with "Produced" — none of it is boilerplate.
    prose = (b"The effect produced by his speech was immense.\n\n"
             b"The old transcribers made many errors in the manuscripts.")
    body = BODY[:3000] + b"\n\n" + prose + b"\n\n" + BODY[3000:]
    out, _ = g.clean(body)
    assert prose in out


def test_mid_book_marker_like_text_does_not_truncate():
    # an END-like line in the first half must not cut the book
    body = b"*** END of the first part ***\n\n" + BODY
    out, _ = g.clean(body)
    assert BODY[-200:] in out


def test_guard_reverts_excessive_stripping():
    # a short "book" that is mostly about e-texts: the paragraph pass would
    # remove most of it, so the guard keeps the marker-cut text unchanged.
    paras = [b"This e-text describes how e-texts were made at the archive."] * 6
    body = b"\n\n".join(paras + [b"A single ordinary sentence remains here."])
    out, st = g.clean(body)
    assert st.guarded
    assert out == body


def test_pick_variants_prefers_utf8_then_8bit():
    ps = [Path(p) for p in ("a/12-8.txt", "a/12.txt", "a/12-0.txt",
                            "b/77.txt", "b/77-8.txt", "c/old10.txt")]
    kept, dropped = g.pick_variants(ps)
    assert {p.name for p in kept} == {"12-0.txt", "77-8.txt", "old10.txt"}
    assert dropped == 3


WORDS = ("which while though quite school night ought weigh whole rough "
         "queen question light thought where whether sought bright through "
         "the and of a to in was he she it that with his her for on").split()


def prose(seed: int, n: int = 30000) -> bytes:
    import random
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n)).encode()


def test_neardup_detects_variants_and_rewrap_not_distinct_books():
    a = prose(1)
    b = (a.replace(b"queen", b"qu\xc3\xa9en")          # accents differ
          .replace(b" the ", b"\nthe "))               # re-wrapped
    b = b"A different front matter page.\n\n" + b    # shifted start
    other = prose(2)
    nd = g.NearDup()
    assert nd.check_add(g.fingerprint(a)) == -1
    assert nd.check_add(g.fingerprint(b)) == 0
    assert nd.check_add(g.fingerprint(other)) == -1
