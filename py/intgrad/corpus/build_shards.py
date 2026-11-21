"""build_shards — dataset builder: raw text sources -> intgrad byte shards.

Output layout (data/corpus-<name>/):
  shard-0000.bin ...   raw UTF-8 documents, concatenated, separator "\\n \\n"
  manifest.jsonl       one record per shard: {"shard": name, "bytes": n,
                       "docs": [{"off","len","src","kind","lang"}]}
  dataset.json         {"version", "created", "separator", "stats"}

Guarantees for the byte-level trainer:
  * payload contains no C0 control bytes except \\t and \\n (all others are
    stripped, \\r is normalized to \\n) -- keeps the control range free for
    in-band markers (plan/001 D7: 0x03 end-of-document, 0x00 padding)
  * exact-duplicate documents dropped (blake2b-128)
  * Project Gutenberg: boilerplate stripped, one encoding variant per book
    id, near-duplicate books dropped (intgrad.corpus.gutenberg)
  * per-document provenance recorded (source path, kind, language)

Sources:
  --gutenberg-tree DIR  recursive *.txt (PG mirror)        kind: gutenberg
  --corpus-dir DIR      one doc per .txt file              kind: textcorpus
  --wiki-defs FILE      "Title ||| definition" line dump   kind: wiki2017
  --wiki-xml LANG=PATH  MediaWiki pages-articles dump      kind: wikixml
                        (bz2 auto-detected; repeatable)
  --max-docs N          stop after N accepted docs (sample runs)
  --jobs N              worker processes for book cleaning (default: all)

Plan-003 sources (intgrad.corpus.ingest; per-doc lic/gated/private/grp):
  --mdbooks DIR         Markdown books, classes from --mdbook-scores  kind: mdbook
  --hf NAME ...         ig-fetch sources from --hf-root               kinds: edu cosmo web
                                                                       math code tool
  --exclude-gated       leave out files fetched under accepted terms

Language filter: --langs en,de,fr,es,it,pt,nl (default); "unk" keeps
unknown-language docs too (function-word scoring on body windows).

Run:  cd py && uv run --extra corpus ig-build-shards --out ../data/corpus-v2 ...
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from intgrad.corpus import gutenberg as pg

SEP = b"\n \n"
# Delete every C0 control except \t/\n: the control range stays reserved
# for in-band markers the trainer may insert.
_DELETE = bytes(i for i in range(0x20) if i not in (0x09, 0x0A))


def sanitize(raw: bytes) -> bytes:
    b = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return b.translate(None, _DELETE)  # C-speed strip of control bytes


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


_RE = {
    "comment": re.compile(r"<!--.*?-->", re.S),
    "pre": re.compile(r"<(nowiki|source|pre|code|syntaxhighlight)\b.*?</\1>", re.S | re.I),
    "ref_self": re.compile(r"<ref\b[^>]*?/>", re.S),
    "ref": re.compile(r"<ref\b.*?</ref>", re.S | re.I),
    "gallery": re.compile(r"<(img|gallery|imagemap|timeline)[^>]*?>.*?</\1>", re.S | re.I),
    "table": re.compile(r"^\s*\{\|.*?^\s*\|\}\s*", re.S | re.M),
    "tmpl": re.compile(r"\{\{[^{}]*\}\}"),
    "brolink": re.compile(
        r"\[\[(?:File|Image|Category|MediaWiki|Special|Talk|Wikipedia|Help"
        r"|Portal|Template|Draft|User|Media|Benutzer|Diskussion|Datei|Kategorie)"
        r":[^\]]*\]\]", re.I),
    "ext_named": re.compile(r"\[(?:https?|ftp)://\S+ +([^\]]+)\]"),
    "ext_bare": re.compile(r"\[(?:https?|ftp)://[^\]]+\]"),
    "link_named": re.compile(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]"),
    "link_plain": re.compile(r"\[\[([^\[\]]*)\]\]"),
    "heading": re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.M),
    "listmark": re.compile(r"^[*#:;\s]+", re.M),
    "magic": re.compile(r"__[A-Za-z]+__"),
    "tags": re.compile(r"</?[a-zA-Z][^>]*>"),
    "straybr": re.compile(r"\[\[|\]\]"),
    "hws": re.compile(r"[ \t]+"),
    "nbs": re.compile(r"\n{3,}"),
}


def strip_wiki(t: str) -> str:
    """MediaWiki markup -> plain prose. v1 stripper: tuned for readability
    of the result, not spec completeness."""
    t = _RE["comment"].sub(" ", t)
    t = _RE["pre"].sub(" ", t)
    t = _RE["ref_self"].sub(" ", t)
    t = _RE["ref"].sub(" ", t)
    t = _RE["gallery"].sub(" ", t)
    t = _RE["table"].sub(" ", t)
    prev = None
    while prev != t:            # innermost-out removal of nested templates
        prev = t
        t = _RE["tmpl"].sub(" ", t)
    t = _RE["brolink"].sub(" ", t)
    t = _RE["ext_named"].sub(r"\1", t)
    t = _RE["ext_bare"].sub(" ", t)
    t = _RE["link_named"].sub(r"\1", t)
    t = _RE["link_plain"].sub(r"\1", t)
    t = _RE["heading"].sub(r"\1", t)
    t = _RE["listmark"].sub("", t)
    t = _RE["magic"].sub(" ", t)
    t = t.replace("'''", "").replace("''", "")
    t = _RE["tags"].sub(" ", t)
    t = _RE["straybr"].sub(" ", t)   # orphan [[ / ]] from malformed nesting
    t = _RE["hws"].sub(" ", t)
    t = _RE["nbs"].sub("\n\n", t)
    return t.strip()


def _is_disambig(title: str, raw: str) -> bool:
    if title.lower().endswith("(disambiguation)"):
        return True
    head = raw[:1500]
    return ("__DISAMBIG" in head.upper()
            or "may refer to" in head.lower()
            or "kann sich beziehen" in head.lower())


def iter_wiki_xml(lang: str, path: str):
    """Streaming MediaWiki dump -> plain-text articles (ns0 only, no
    redirects, no disambiguation pages). XML entities decoded by parser.
    src field embeds the language: 'LANG|||file:title'."""
    import bz2
    from lxml import etree

    with open(path, "rb") as probe:
        is_bz2 = probe.read(3) == b"BZh"
    opener = bz2.open if is_bz2 else open
    ns = "{http://www.mediawiki.org/xml/export-0.10/}"
    with opener(path, "rb") as fh:  # type: ignore[operator]
        context = etree.iterparse(fh, events=("end",), tag=ns + "page",
                                  huge_tree=True, recover=True)
        for _, page in context:
            ns_el = page.find(ns + "ns")
            title_el = page.find(ns + "title")
            rev = page.find(ns + "revision")
            text_el = None if rev is None else rev.find(ns + "text")
            raw = (text_el.text if text_el is not None else None) or ""
            title = (title_el.text if title_el is not None else "") or ""
            skip = (ns_el is None or (ns_el.text or "0").strip() != "0"
                    or page.find(ns + "redirect") is not None)
            page.clear()
            while page.getprevious() is not None:  # free memory as we go
                del page.getparent()[0]
            if skip or _is_disambig(title, raw):
                continue
            prose = strip_wiki(raw)
            if not prose:
                continue
            doc = sanitize(prose.encode("utf-8"))
            yield doc, f"{lang}|||{Path(path).name}:{title}", "wikixml"


def _clean_book(fp: Path):
    """Worker: read + clean + sanitize + fingerprint one PG file."""
    try:
        raw = fp.read_bytes()
    except OSError:
        return None
    doc, st = pg.clean(raw)
    doc = sanitize(doc).strip()
    return doc, str(fp), st, pg.fingerprint(doc), detect_lang(doc)


def iter_gutenberg(root: Path, jobs: int, stats: dict):
    files, nvar = pg.pick_variants(sorted(root.rglob("*.txt")))
    g = stats.setdefault("gutenberg", {
        "files_variant_dropped": nvar, "files": len(files), "neardup": 0,
        "guarded": 0, "header_bytes": 0, "footer_bytes": 0,
        "para_bytes": 0, "paras": 0, "inline_notes": 0})
    nd = pg.NearDup()
    with mp.Pool(jobs or None) as pool:
        for res in pool.imap(_clean_book, files, chunksize=8):
            if res is None:
                continue
            doc, src, st, fp, lang = res
            g["guarded"] += st.guarded
            g["header_bytes"] += st.header
            g["footer_bytes"] += st.footer
            g["para_bytes"] += st.para_bytes
            g["paras"] += st.paras
            g["inline_notes"] += st.inline
            if not doc:
                continue
            if nd.check_add(fp) >= 0:
                g["neardup"] += 1
                continue
            yield doc, src, "gutenberg", lang


def iter_docs(args, stats: dict):
    """Yields (doc, src, kind, lang or None) or, for plan-003 sources,
    (doc, src, kind, lang, meta) with meta merged into the manifest record."""
    for spec in args.wiki_xml:
        lang, _, path = spec.partition("=")
        for doc, src, kind in iter_wiki_xml(lang, path):
            yield doc, src, kind, lang
    if args.wiki_defs:
        p = Path(args.wiki_defs)
        with open(p, "rb") as f:
            for lineno, line in enumerate(f):
                d = sanitize(doc_from_wiki_line(line) or b"")
                if d:
                    yield d, f"{p.name}#{lineno}", "wiki2017", "en"
    if args.gutenberg_tree:
        yield from iter_gutenberg(Path(args.gutenberg_tree), args.jobs, stats)
    if args.corpus_dir:
        d = Path(args.corpus_dir)
        for fp in sorted(d.iterdir()):
            if fp.is_file() and fp.suffix == ".txt":
                yield sanitize(pg.to_utf8(fp.read_bytes())), str(fp), "textcorpus", None
    if args.mdbooks:
        from intgrad.corpus import ingest
        yield from ingest.iter_mdbooks(Path(args.mdbooks).expanduser(), Path(args.mdbook_scores),
                                       set(args.mdbook_classes.split(",")),
                                       set(args.langs.split(",")), args.jobs)
    if args.hf:
        from intgrad.corpus import ingest
        yield from ingest.iter_hf(Path(args.hf_root), set(args.hf), args.exclude_gated, stats)


def verify(out: Path) -> dict:
    """Re-open every written shard and validate manifest offsets: each doc
    slice must be non-empty, not start with the separator, and docs must be
    exactly SEP-joined with the last one ending at EOF."""
    problems, docs = 0, 0
    with open(out / "manifest.jsonl", encoding="utf-8") as mf:
        for line in mf:
            rec = json.loads(line)
            data = (out / rec["shard"]).read_bytes()
            if len(data) != rec["bytes"]:
                print(f"VERIFY FAIL {rec['shard']}: size mismatch",
                      file=sys.stderr)
                problems += 1
            dlist = rec["docs"]
            for i, d in enumerate(dlist):
                docs += 1
                doc = data[d["off"]:d["off"] + d["len"]]
                if not doc or doc.startswith(SEP) or doc[:1] in (b"\n", b" "):
                    print(f"VERIFY FAIL {rec['shard']}#{i}: bad doc start",
                          file=sys.stderr)
                    problems += 1
                end = d["off"] + d["len"]
                if i + 1 < len(dlist):
                    nxt = dlist[i + 1]
                    if data[end:end + len(SEP)] != SEP or nxt["off"] != end + len(SEP):
                        print(f"VERIFY FAIL {rec['shard']}#{i}: gap",
                              file=sys.stderr)
                        problems += 1
                elif end != len(data):
                    print(f"VERIFY FAIL {rec['shard']}: trailing gap",
                          file=sys.stderr)
                    problems += 1
    return {"docs_verified": docs, "problems": problems}


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text("")  # idempotent re-runs
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

    allowed = set(args.langs.split(",")) | {"zxx"}     # zxx: no natural language (code)
    for item in iter_docs(args, stats):
        doc, src, kind, lang = item[:4]
        meta = item[4] if len(item) > 4 else {}
        stats["docs_in"] += 1
        doc = doc.strip()  # invariant: docs never start/end with whitespace,
        if len(doc) < args.min_doc_bytes:  # so SEP boundaries are unambiguous
            stats["docs_short"] += 1
            continue
        if lang is None:
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
        if buf:
            buf += SEP
        off = len(buf)  # AFTER the separator: off points at the doc itself
        buf += doc
        docs_in_shard.append({"off": off, "len": len(doc), "src": src,
                              "kind": kind, "lang": lang, **meta})
        stats["docs_out"] += 1
        stats["bytes_out"] += len(doc)
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        stats.setdefault("by_lang", {})
        stats["by_lang"][lang] = stats["by_lang"].get(lang, 0) + 1
        if args.max_docs and stats["docs_out"] >= args.max_docs:
            break
    flush_shard()

    stats["verify"] = verify(out)
    (out / "dataset.json").write_text(json.dumps(
        {"version": 1, "created": int(time.time()),
         "separator": SEP.decode(), "stats": stats}, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    if stats["verify"]["problems"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gutenberg-tree")
    ap.add_argument("--corpus-dir")
    ap.add_argument("--wiki-defs")
    ap.add_argument("--wiki-xml", action="append", default=[],
                    metavar="LANG=PATH",
                    help="MediaWiki pages-articles dump (repeatable)")
    ap.add_argument("--mdbooks", help="Markdown book tree (plan 003 §2.1), kind mdbook")
    ap.add_argument("--mdbook-scores", default=str(Path(__file__).resolve().parents[3]
                                                   / "data" / "mdbooks" / "scores.jsonl"))
    ap.add_argument("--mdbook-classes", default="keep",
                    help="comma list of ig-mdbook-score classes to include")
    ap.add_argument("--hf", nargs="*", default=[], metavar="NAME",
                    help="ig-fetch source names to include (plan 003 §2.3)")
    ap.add_argument("--hf-root", default=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "hf"))
    ap.add_argument("--exclude-gated", action="store_true",
                    help="skip files downloaded under accepted terms (open rebuild, D10)")
    ap.add_argument("--langs", default="en,de,fr,es,it,pt,nl")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--shard-bytes", type=int, default=64 << 20)
    ap.add_argument("--min-doc-bytes", type=int, default=64)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
