"""Project Gutenberg cleanup: boilerplate stripping, variant selection,
near-duplicate fingerprints.

PG files wrap the book in material that is not part of it: license
headers/footers (several historical formats), production credits
("Produced by …", "E-text prepared by …", proofreading-team and scan-source
notes), transcriber's notes, pointers to HTML versions and URLs. Front
matter that belongs to the BOOK (title page, contents, dedication, preface)
is kept.

Pipeline (`clean`):
  1. encoding -> UTF-8, CRLF -> LF
  2. marker cut: last START-type marker in the head window (modern
     "*** START OF … ***", possibly wrapped over lines; old "*END*THE SMALL
     PRINT!" license end), first END-type marker in the second half
  3. paragraph pass in the head zone and tail zone only: drop paragraphs
     that score as boilerplate; bracketed multi-paragraph transcriber notes
     are removed as a unit; a tail "Transcriber's Note(s)" section is
     removed to the end; short paragraphs sandwiched between boilerplate
     are removed too; decoration-only paragraphs at the edges are dropped
  4. guard: if the paragraph pass would remove more than GUARD_FRAC of the
     marker-cut text, it is reverted (counted in stats) — precision first.

Variant selection (`pick_variants`): PG stores many books as N.txt (ASCII),
N-8.txt (8-bit) and N-0.txt (UTF-8). Exact-hash dedup cannot see that these
are the same book; we keep one per id, preferring -0 > -8 > plain.

Near duplicates (`fingerprint`, `NearDup`): content-defined shingle
sampling over an accent-folded, letters-only view of the text, robust to
encoding variants, re-wrapping and different front matter.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- encoding


def to_utf8(raw: bytes) -> bytes:
    """Normalize mixed PG encodings (ascii/latin-1/cp1252/utf-8) to UTF-8."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        raw.decode("utf-8")
        return raw
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace").encode("utf-8")


# ---------------------------------------------------------------- markers

_I = re.I
_START_RES = [
    # modern: *** START OF THE PROJECT GUTENBERG EBOOK TITLE *** (may wrap)
    re.compile(rb"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^*]{0,600}\*{3}", _I),
    re.compile(rb"^[ \t]*\*{3}\s*START\b[^\n]*", _I | re.M),
    # old license block end: *END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*
    re.compile(rb"\*\s*END\s*\*?\s*THE SMALL PRINT[^\n]*", _I),
    re.compile(rb"^[ \t]*Start of (?:the|this) Project Gutenberg[^\n]*", _I | re.M),
]
_END_RES = [
    re.compile(rb"^[ \t]*\*{0,3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG", _I | re.M),
    re.compile(rb"^[ \t]*End of (?:the |this )?Project Gutenberg", _I | re.M),
    re.compile(rb"^[ \t]*End of (?:the |this )?(?:Project Gutenberg )?E-?(?:text|book)\b", _I | re.M),
    re.compile(rb"^[ \t]*\*{3}\s*END\b", _I | re.M),
    re.compile(rb"^[ \t]*\*+\s*This file should be named", _I | re.M),
]

HEAD_WINDOW = 64 << 10  # START markers are searched in min(this, len/2)


def cut_markers(t: bytes) -> tuple[bytes, int, int]:
    """Return (body, header_bytes_cut, footer_bytes_cut)."""
    win = min(HEAD_WINDOW, len(t) // 2)
    lo = 0
    for rx in _START_RES:
        for m in rx.finditer(t, 0, win):
            if m.end() > lo:
                lo = m.end()
    if lo:
        eol = t.find(b"\n", lo)
        lo = len(t) if eol < 0 else eol + 1
    hi = len(t)
    half = lo + (len(t) - lo) // 2
    for rx in _END_RES:
        m = rx.search(t, half)
        if m and m.start() < hi:
            hi = m.start()
    if hi < lo:
        return t, 0, 0
    return t[lo:hi], lo, len(t) - hi


# ---------------------------------------------------------------- paragraphs

_STRONG = re.compile(
    rb"gutenberg|\bpgdp\b|distributed\s+proof|\be-?texts?\b|\be-?books?\b"
    rb"|\A\s*\[?\s*(?:produced|prepared|transcribed|scanned|digiti[sz]ed|proofread|"
    rb"converted|updated|edited for project)\b[^\n]{0,80}\bby\b"
    rb"|\b(?:this|the) (?:file|text|book|edition) was (?:produced|prepared|created|"
    rb"scanned|made|first posted)"
    rb"|transcriber(?:'|\xe2\x80\x99)?s?(?:'|\xe2\x80\x99)?\s*notes?|note (?:from|by) the transcriber"
    rb"|internet archive|archive\.org|google books|hathitrust|bibliotheque nationale"
    rb"|biblioth\xc3\xa8que nationale|\bpage images\b|\bhtml version\b|illustrated html"
    rb"|images? (?:generously )?(?:made )?available|https?://|\bwww\.|\bftp\b"
    rb"|small print|copyright laws are changing|for the use of anyone anywhere"
    rb"|\b(?:p\.|page)\s*\d+\b[^\n]{0,120}\b(?:changed|corrected|replaced|amended|"
    rb"was printed|transposed|supplied)\b"
    rb"|\bdonations?\b[^\n]{0,60}\b(?:foundation|literary archive)",
    re.I)
_WEAK = re.compile(
    rb"\bcopyright\b|\bpublic domain\b|redistribut|\blicen[cs]e\b|\btrademark\b"
    rb"|\bthis header\b|\bemail\b|@[a-z0-9-]+\.[a-z]|\bascii\b|\bunicode\b|\blatin-1\b"
    rb"|\butf-?8\b|\bdigital\b|\bonline\b|\bdownload|\bindex(?:es)? of\b"
    rb"|\bchanged to\b|\bcorrected\b|\btypographical\b|\bspelling\b|\bhyphenat",
    re.I)
_DECOR = re.compile(rb"^(?:[\s*\-=_~#.]|\xe2\x80\xa2)*$")
_APOS = rb"(?:'|\xe2\x80\x99)?"  # ASCII or typographic apostrophe
_TNOTE = re.compile(
    rb"^\s*\[?\s*(?:major\s+)?transcriber" + _APOS + rb"s?" + _APOS
    + rb"\s*(?:end)?(?:notes?|changes|corrections)\b",
    re.I)
# Anywhere in the book: a bracketed inline transcriber's note (no nesting),
# and the World Library copyright notice repeated between Shakespeare plays.
_INLINE_NOTES = re.compile(
    rb"[ \t]*\[\s*transcriber" + _APOS + rb"s?" + _APOS
    + rb"\s*(?:end)?notes?\s*:?[^\[\]]{0,1500}\]"
    rb"|<<\s*THIS ELECTRONIC VERSION[^>]{0,2000}>>",
    re.I)
_PARA_SPLIT = re.compile(rb"\n[ \t]*\n")

HEAD_ZONE = 24 << 10   # paragraph pass covers min(this, 25%) at each end
TAIL_ZONE = 32 << 10
GUARD_FRAC = 0.30
TNOTE_HEADING = 80     # a tail note *heading* line, not an inline note
TNOTE_SECTION = 12 << 10  # ... removed to the end only if this close to it


def _mostly_lower(p: bytes) -> bool:
    """Sentence-like (license line) rather than a title (often all caps)."""
    letters = [c for c in p if 65 <= c <= 90 or 97 <= c <= 122]
    return bool(letters) and sum(c >= 97 for c in letters) >= 0.6 * len(letters)


def is_boilerplate(p: bytes) -> bool:
    if _STRONG.search(p):
        return True
    return len({m.lower() for m in _WEAK.findall(p)}) >= 2  # distinct signals


def _paragraphs(t: bytes) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _PARA_SPLIT.finditer(t):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(t)))
    return spans


def strip_paragraphs(t: bytes) -> tuple[bytes, int]:
    """Drop boilerplate paragraphs in the head and tail zones.
    Returns (text, n_paragraphs_removed)."""
    spans = _paragraphs(t)
    n = len(spans)
    head_lim = min(HEAD_ZONE, len(t) // 4)
    tail_lim = len(t) - min(TAIL_ZONE, len(t) // 4)
    in_head = [s < head_lim for s, _ in spans]
    in_tail = [e > tail_lim and not in_head[i] for i, (_, e) in enumerate(spans)]
    drop = [False] * n
    bp = [False] * n
    for i, (s, e) in enumerate(spans):
        if in_head[i] or in_tail[i]:
            bp[i] = is_boilerplate(t[s:e])

    i = 0
    while i < n:
        if not (in_head[i] or in_tail[i]):
            s0, e0 = spans[i]
            if e0 - s0 <= 1500 and _TNOTE.match(t[s0:e0]):
                drop[i] = True          # mid-book "Transcriber's note: ..."
            i += 1
            continue
        p = t[spans[i][0]:spans[i][1]]
        if _TNOTE.match(p):
            closed = p.lstrip().startswith(b"[") and p.rstrip().endswith(b"]")
            if (in_tail[i] and not p.lstrip().startswith(b"[")
                    and len(p.strip()) <= TNOTE_HEADING
                    and len(t) - spans[i][0] <= TNOTE_SECTION):
                for k in range(i, n):           # "Transcriber's Notes" section
                    drop[k] = True              # at the very end: to the end
                break
            if closed or not p.lstrip().startswith(b"["):  # single paragraph
                drop[i] = True                  # the note paragraph itself
                i += 1
                continue
            k = i                               # unclosed "[Transcriber's Note:":
            while k < n and k - i < 30:         # drop through the closing "]"
                drop[k] = True
                if t[spans[k][0]:spans[k][1]].rstrip().endswith(b"]"):
                    break
                k += 1
            i = k + 1
            continue
        if bp[i]:
            drop[i] = True
        i += 1

    # short paragraphs sandwiched between dropped boilerplate (license lines)
    for i in range(1, n - 1):
        if (not drop[i] and (in_head[i] or in_tail[i]) and bp[i - 1] and bp[i + 1]
                and spans[i][1] - spans[i][0] < 300
                and _mostly_lower(t[spans[i][0]:spans[i][1]])):
            drop[i] = True
    # decoration-only paragraphs left dangling at the edges
    for order in (range(n), range(n - 1, -1, -1)):
        for i in order:
            if drop[i]:
                continue
            if _DECOR.match(t[spans[i][0]:spans[i][1]]):
                drop[i] = True
            else:
                break

    kept = [t[s:e] for i, (s, e) in enumerate(spans) if not drop[i]]
    return b"\n\n".join(kept), sum(drop)


@dataclass
class CleanStats:
    header: int = 0
    footer: int = 0
    para_bytes: int = 0
    paras: int = 0
    inline: int = 0
    guarded: bool = False


def clean(raw: bytes) -> tuple[bytes, CleanStats]:
    """Full PG cleanup of one file's bytes -> (book text, stats)."""
    t = to_utf8(raw).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    st = CleanStats()
    body, st.header, st.footer = cut_markers(t)
    stripped, st.paras = strip_paragraphs(body)
    stripped, st.inline = _INLINE_NOTES.subn(b"", stripped)
    removed = len(body) - len(stripped)
    if removed > GUARD_FRAC * max(len(body), 1):
        st.guarded = True
        st.paras = st.inline = 0
        stripped = body
        removed = 0
    st.para_bytes = removed
    return stripped.strip(), st


# ---------------------------------------------------------------- variants

_VARIANT = re.compile(r"^(\d+)(?:-(\d))?\.txt$")
_PREF = {"0": 0, "8": 1, None: 2}  # UTF-8 > 8-bit > ASCII


def pick_variants(paths: list[Path]) -> tuple[list[Path], int]:
    """Keep one file per numeric PG id (-0 > -8 > plain). Non-numeric (old
    etextNN) names pass through. Returns (kept in sorted order, n_dropped)."""
    best: dict[str, tuple[int, Path]] = {}
    other: list[Path] = []
    for p in paths:
        m = _VARIANT.match(p.name)
        if not m or m.group(2) not in _PREF:
            other.append(p)
            continue
        rank = _PREF[m.group(2)]
        cur = best.get(m.group(1))
        if cur is None or rank < cur[0]:
            best[m.group(1)] = (rank, p)
    kept = sorted(other + [p for _, p in best.values()])
    return kept, len(paths) - len(kept)


# ---------------------------------------------------------------- near-dups

_ANCHOR = re.compile(rb"qu|sch|wh|ij|gh")
_NONALPHA = re.compile(rb"[^a-z]+")
SHINGLE = 40
SAMPLE_MASK = 7          # keep shingles whose hash & MASK == 0  (~1/8)
MIN_SHINGLES = 16


def fingerprint(doc: bytes) -> list[int]:
    """Sampled shingle hashes of an accent-folded, letters-only view.
    Content-defined (anchored on letter n-grams), so it is stable under
    re-wrapping, encoding variants and shifted front matter."""
    s = unicodedata.normalize("NFKD", doc.decode("utf-8", "replace"))
    s = _NONALPHA.sub(b"", s.encode("ascii", "ignore").lower())
    out = []
    for m in _ANCHOR.finditer(s):
        h = zlib.crc32(s[m.start():m.start() + SHINGLE])
        if h & SAMPLE_MASK == 0:
            out.append(h)
    return out


@dataclass
class NearDup:
    """First-seen-wins near-duplicate filter: a doc is a duplicate if at
    least `thresh` of its sampled shingles belong to ONE earlier doc
    (containment, so an excerpt of a kept book is dropped too)."""
    thresh: float = 0.5
    owner: dict[int, int] = field(default_factory=dict)
    n: int = 0

    def check_add(self, fp: list[int]) -> int:
        """Return the id of the earlier doc this one duplicates, or -1 (and
        register it)."""
        if len(fp) >= MIN_SHINGLES:
            hits = Counter(self.owner[h] for h in fp if h in self.owner)
            if hits:
                did, c = hits.most_common(1)[0]
                if c >= self.thresh * len(fp):
                    return did
        did = self.n
        self.n += 1
        for h in fp:
            self.owner.setdefault(h, did)
        return -1
