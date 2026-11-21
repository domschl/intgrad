"""mdbook — Markdown books: suitability score and class per book (plan 003 §2.1, D6).

    uv run --project py ig-mdbook-score [--src ~/Docs/KnrsData/MarkdownBooks]
                                        [--out data/mdbooks/scores.jsonl]

Each book gets structural statistics (no text is stored):
  prose    share of bytes in prose paragraphs (blank-line separated blocks,
           ≥ 120 bytes, mostly letters; hard-wrapped lines are joined)
  table    share of bytes in table blocks ("|" rows)
  code     share of bytes inside ``` fences
  math     share of bytes in display-math or LaTeX-dense blocks
  list     share of bytes in list blocks
  short    share of blocks shorter than 40 bytes (headings, page numbers)
  wrapped  share of blocks that are hard-wrapped prose (cleaner unwraps them)
  digits   share of digit bytes
  fffd     U+FFFD replacement characters per MB
  latex    $$ / \\( / \\begin{ markers per MB
  lang     en / de / ... by function-word count (data metadata only)

and one class:
  poor       conversion damage: fffd > 50/MB, or table > 0.30, or
             prose + code + math < 0.25
  reference  reference-like layout: short > 0.55, or digits > 0.08, or list > 0.45
  keep       otherwise

The output (data/ only: it names private books) is one JSON line per book:
{"path", "topic", "group", "bytes", <stats>, "class"}. Thresholds are first guesses,
to be checked against counts per class and user spot checks.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from collections import Counter
from pathlib import Path

from intgrad.corpus.build_shards import LANGWORDS

# topic folder -> group, for mixture weights (folders not listed: "general")
GROUPS = {
    "stem": ["AI", "Astronomy", "Biology", "Computer", "Electronics", "Genomics", "Geology",
             "Health", "Mathematics", "Neuroscience", "Physics", "Programming"],
    "humanities": ["Anthropology", "Archaeology", "Art", "History", "Language", "Music",
                   "Philosophy", "Photography", "Psychologie"],
    "narrative": ["Comic", "Epos", "Fiction", "Literatur", "Literature", "Science Fiction"],
    "religion": ["Dharma", "Lotsawahouse", "Religion"],
    "practical": ["Games", "Kochen", "Manuals", "München", "Practicebooks", "Ratgeber"],
}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}

ROOT = Path(__file__).resolve().parents[3]
_LETTER = re.compile(rb"[A-Za-z\x80-\xff]")
_LATEX = re.compile(rb"\$\$|\\\(|\\begin\{")
_LIST = re.compile(rb"^\s*(?:[-*+]|\d{1,3}[.)])\s")
_WORD = re.compile(r"[^\W\d_]+")
_LW = {k: set(v) for k, v in LANGWORDS.items()}

THRESH = {"fffd": 50.0, "table": 0.30, "prose": 0.25, "short": 0.55, "digits": 0.08, "list": 0.45}


def lang_of(b: bytes, window: int = 400_000) -> str:
    words = Counter(w.lower() for w in _WORD.findall(b[:window].decode("utf-8", "ignore")))
    score = {k: sum(words[w] for w in ws) for k, ws in _LW.items()}
    best = max(score, key=score.get)
    return best if score[best] > 0 else "unk"


def blocks(b: bytes):
    """Paragraph blocks (blank-line separated) outside code fences, plus the
    fenced code bytes. Hard-wrapped paragraphs count as one block."""
    out, cur, code, fence = [], [], 0, False
    for line in b.split(b"\n"):
        if line.lstrip().startswith(b"```"):
            fence = not fence
            code += len(line) + 1
            if cur:
                out.append(cur)
                cur = []
            continue
        if fence:
            code += len(line) + 1
        elif line.strip():
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out, code


def stats(b: bytes) -> dict:
    n = max(len(b), 1)
    bl, code = blocks(b)
    prose = table = lst = short = wrapped = math = 0
    for lines in bl:
        L = sum(len(x) + 1 for x in lines)
        first = lines[0].lstrip()
        if first.startswith(b"|"):
            table += L
        elif _LIST.match(lines[0]):
            lst += L
        elif first.startswith(b"#"):
            pass
        elif first.startswith(b"$$") or sum(x.count(b"\\") for x in lines) >= 3:
            math += L
        else:
            text = b" ".join(x.strip() for x in lines)
            if len(text) >= 120 and len(_LETTER.findall(text)) > 0.6 * len(text):
                prose += L
                wrapped += len(lines) > 1
        if L < 40:
            short += 1
    mb = n / 1e6
    return {
        "prose": round(prose / n, 4), "math": round(math / n, 4), "table": round(table / n, 4), "code": round(code / n, 4),
        "list": round(lst / n, 4), "short": round(short / max(len(bl), 1), 4),
        "wrapped": round(wrapped / max(len(bl), 1), 4),
        "digits": round(sum(b.count(bytes([c])) for c in range(48, 58)) / n, 4),
        "fffd": round(b.count("\ufffd".encode()) / mb, 2),
        "latex": round(len(_LATEX.findall(b)) / mb, 2),
        "lang": lang_of(b),
    }


def classify(s: dict) -> str:
    t = THRESH
    content = s["prose"] + s["code"] + s["math"]          # code and maths are wanted content
    if s["fffd"] > t["fffd"] or s["table"] > t["table"] or content < t["prose"]:
        return "poor"
    if s["short"] > t["short"] or s["digits"] > t["digits"] or s["list"] > t["list"]:
        return "reference"
    return "keep"


def _one(args):
    path, src = args
    b = Path(path).read_bytes()
    s = stats(b)
    rel = os.path.relpath(path, src)
    topic = rel.split(os.sep)[0]
    return {"path": rel, "topic": topic, "group": GROUP_OF.get(topic, "general"), "bytes": len(b),
            **s, "class": classify(s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("~/Docs/KnrsData/MarkdownBooks").expanduser())
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "mdbooks" / "scores.jsonl")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    files = sorted(str(p) for p in a.src.rglob("*.md"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(a.jobs) as pool, open(a.out, "w") as f:
        rows = list(pool.imap(_one, [(p, a.src) for p in files], chunksize=8))
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = Counter()
    for r in rows:
        by[(r["class"], "n")] += 1
        by[(r["class"], "MB")] += r["bytes"] / 1e6
    for c in ("keep", "reference", "poor"):
        print(f"{c:9s} {by[(c, 'n')]:5d} books {by[(c, 'MB')]:7.0f} MB")
    g = Counter()
    for r in rows:
        g[(r["group"], r["lang"] if r["lang"] in ("en", "de") else "other", r["class"])] += r["bytes"] / 1e6
    for grp in [*GROUPS, "general"]:
        print(f"{grp:10s}", {f"{l}/{c}": round(v) for (gg, l, c), v in sorted(g.items()) if gg == grp and v >= 1})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
