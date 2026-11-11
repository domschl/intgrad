# /// script
# requires-python = ">=3.11"
# ///
"""build_shards.py — dataset builder: raw text sources -> intgrad byte shards.

Output layout (data/corpus-<name>/):
  shard-0000.bin ...   raw UTF-8 documents, concatenated, separator "\\n \\n"
  manifest.jsonl       one record per shard: {"shard": name, "bytes": n,
                       "docs": [{"off","len","src"}]}
  dataset.json         {"version", "created", "separator", "stats"}

Guarantees for the byte-level trainer + control-byte protocol (D2):
  * payload never contains ESC (0x00) or other protocol marker bytes
    (they are stripped; \\t kept, \\r normalized to \\n)
  * exact-duplicate documents dropped (blake2b-128)
  * per-document provenance recorded (source path / kind)

Sources (add more adapters as needed):
  --gutenberg-dir DIR   one book per .txt file            src kind: gutenberg
  --gutenberg-tree DIR  recursive *.txt (PG mirror), encoding-normalized
                        + language-detected               src kind: gutenberg
  --corpus-dir DIR      one doc per .txt file             src kind: textcorpus
  --wiki-defs FILE      "Title ||| definition" line dump  src kind: wiki2017

Language filter: --langs en,de,fr,es,it,pt,nl (default); "unk" keeps
unknown-language docs too. Detection = function-word scoring over the first
4 KB, good enough as a v1 gate for the Latin-family curriculum (D4).

Run with uv:   uv run tools/build_shards.py --out data/corpus-v0 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

SEP = b"\n \n"
# All protocol marker bytes (0x00, 0x0B, 0x0C, 0x1C..0x1F) are C0 controls;
# we delete every C0 control except \t/\n, which covers them all.
_DELETE = bytes(i for i in range(0x20) if i not in (0x09, 0x0A))


def sanitize(raw: bytes) -> bytes:
    b = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return b.translate(None, _DELETE)  # C-speed strip of protocol/control bytes


PG_STARTS = (b"*** START OF", b"***START OF", b"*** START",
             b"Start of this Project Gutenberg", b"Start of The Project Gutenberg")
PG_ENDS = (b"*** END OF", b"***END OF", b"*** END",
           b"End of this Project Gutenberg", b"End of The Project Gutenberg")


def strip_gutenberg_wrapper(raw: bytes) -> bytes:
    """Cut Project Gutenberg boilerplate header/footer when present."""
    window = raw[:8192]
    lo = -1
    for marker in PG_STARTS:
        i = window.rfind(marker)
        if i >= 0:
            lo = max(lo, i)
    if lo >= 0:
        eol = raw.find(b"\n", lo)
        if eol > 0:
            raw = raw[eol + 1:]
    hi = -1
    for marker in PG_ENDS:
        i = raw.find(marker)
        if i >= 0 and (hi < 0 or i < hi):
            hi = i
    if hi >= 0:
        raw = raw[:hi]
    return raw


def to_utf8(raw: bytes) -> bytes:
    """Normalize mixed PG encodings (ascii/latin-1/cp1252/utf-8) to UTF-8."""
    try:
        raw.decode("utf-8")
        return raw
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace").encode("utf-8")


# Function-word profiles for the Latin-family curriculum (D4).
# Distinctive words carry the discrimination; shared words (de, la, e, ...) merely add noise.
LANGWORDS = {
    "en": ["the", "of", "and", "to", "a", "in", "is", "it", "you", "that",
           "he", "was", "for", "with", "his", "her", "all"],
    "de": ["und", "ist", "nicht", "der", "die", "das", "ein", "eine", "mit",
           "sie", "des", "auch", "als", "von", "den", "dem", "aber"],
    "fr": ["le", "les", "et", "une", "est", "dans", "pour", "sur", "que",
           "qui", "avec", "ces", "des", "au", "aux", "très", "pas"],
    "es": ["que", "el", "los", "las", "por", "una", "para", "con", "no",
           "su", "se", "es", "al", "como", "más", "pero"],
    "it": ["che", "il", "del", "della", "sono", "come", "ma", "più",
           "questa", "quello", "gli", "alla", "anche", "ogni"],
    "pt": ["não", "do", "da", "em", "um", "uma", "para", "com", "as",
           "os", "seu", "sua", "está", "também", "isso"],
    "nl": ["het", "van", "dat", "op", "te", "met", "niet", "mijn",
           "zijn", "maar", "als", "voor", "er", "nog"],
}


def detect_lang(text: bytes) -> str:
    """Score function words from body windows at 25/50/75% of the document.
    The head is deliberately skipped: PG production credits are English
    regardless of book language, and short books are all body anyway."""
    n = len(text)
    toks: list[str] = []
    fracs = (0.0, 0.25, 0.5, 0.75) if n < 16384 else (0.25, 0.5, 0.75)
    for frac in fracs:
        lo = int(n * frac)
        w = text[lo:lo + 2048].lower().split()
        toks.extend(t.decode("utf-8", "ignore") for t in w)
    if len(toks) < 100:
        return "unk"
    scores = {}
    for lang, profile in LANGWORDS.items():
        pf = set(profile)
        scores[lang] = sum(1 for t in toks if t in pf)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, bs = ranked[0]
    second = ranked[1][1]
    if bs >= 10 and bs >= second * 1.2:
        return best
    return "unk"


def doc_from_wiki_line(line: bytes) -> bytes | None:
    title, sep, body = line.partition(b" ||| ")
    if not sep:
        return None
    body = body.strip()
    if not body:
        return None
    if body.lower().startswith(title.strip().lower()):
        return body  # lead already mentions the title; keep prose clean
    return title.strip() + b": " + body


def iter_gutenberg(root: Path):
    for fp in sorted(root.rglob("*.txt")):
        try:
            raw = fp.read_bytes()
        except OSError:
            continue
        doc = sanitize(strip_gutenberg_wrapper(to_utf8(raw)))
        if doc:
            yield doc, str(fp), "gutenberg"


def iter_docs(args):
    if args.wiki_defs:
        p = Path(args.wiki_defs)
        with open(p, "rb") as f:
            for lineno, line in enumerate(f):
                d = sanitize(doc_from_wiki_line(line) or b"")
                if d:
                    yield d, f"{p.name}#{lineno}", "wiki2017"
    for d in [Path(args.gutenberg_dir)] if args.gutenberg_dir else []:
        for fp in sorted(d.glob("*.txt")):
            yield sanitize(strip_gutenberg_wrapper(to_utf8(fp.read_bytes()))), str(fp), "gutenberg"
    if args.gutenberg_tree:
        yield from iter_gutenberg(Path(args.gutenberg_tree))
    if args.corpus_dir:
        d = Path(args.corpus_dir)
        for fp in sorted(d.iterdir()):
            if fp.is_file() and fp.suffix == ".txt":
                yield sanitize(to_utf8(fp.read_bytes())), str(fp), "textcorpus"


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_i = 0
    buf = bytearray()
    docs_in_shard: list[dict] = []
    seen: set[bytes] = set()
    stats = {"docs_in": 0, "docs_dup": 0, "docs_short": 0, "docs_out": 0,
             "bytes_out": 0, "by_kind": {}}

    def flush_shard():
        nonlocal shard_i, buf, docs_in_shard
        if not buf:
            return
        name = f"shard-{shard_i:04d}.bin"
        (out / name).write_bytes(bytes(buf))
        with open(out / "manifest.jsonl", "a", encoding="utf-8") as mf:
            mf.write(json.dumps({"shard": name, "bytes": len(buf),
                                 "docs": docs_in_shard}) + "\n")
        print(f"  wrote {name} ({len(buf)/1e6:.1f} MB, "
              f"{len(docs_in_shard)} docs)", file=sys.stderr)
        shard_i += 1
        buf = bytearray()
        docs_in_shard = []

    allowed = set(args.langs.split(","))
    for doc, src, kind in iter_docs(args):
        stats["docs_in"] += 1
        if len(doc) < args.min_doc_bytes:
            stats["docs_short"] += 1
            continue
        if kind == "wiki2017":
            lang = "en"  # English Wikipedia 2017 definitions dump
        else:
            lang = detect_lang(doc)
        if lang not in allowed:
            stats["docs_langskip"] = stats.get("docs_langskip", 0) + 1
            continue
        h = hashlib.blake2b(doc, digest_size=16).digest()
        if h in seen:
            stats["docs_dup"] += 1
            continue
        seen.add(h)
        if len(buf) + len(SEP) + len(doc) > args.shard_bytes and buf:
            flush_shard()
        off = len(buf)
        if buf:
            buf += SEP
        buf += doc
        docs_in_shard.append({"off": off, "len": len(doc), "src": src,
                              "lang": lang})
        stats["docs_out"] += 1
        stats["bytes_out"] += len(doc)
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        stats.setdefault("by_lang", {})
        stats["by_lang"][lang] = stats["by_lang"].get(lang, 0) + 1
    flush_shard()

    (out / "dataset.json").write_text(json.dumps(
        {"version": 0, "created": int(time.time()),
         "separator": SEP.decode(), "stats": stats}, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gutenberg-dir")
    ap.add_argument("--gutenberg-tree")
    ap.add_argument("--corpus-dir")
    ap.add_argument("--wiki-defs")
    ap.add_argument("--langs", default="en,de,fr,es,it,pt,nl")
    ap.add_argument("--shard-bytes", type=int, default=64 << 20)
    ap.add_argument("--min-doc-bytes", type=int, default=64)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
