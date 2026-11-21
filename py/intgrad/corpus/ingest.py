"""ingest — plan-003 sources -> documents for build_shards (C0a).

Each iterator yields (doc bytes, src, kind, lang, meta). meta goes into the
manifest record per document:
    lic      license (or license class) of this document
    gated    downloaded under accepted terms (ig-fetch registry)
    private  proprietary local source (never distributed)
    grp      split group key: book, repository or conversation; documents
             sharing it share a split bucket (split-v2, plan 003 D3)
    group    Markdown books only: topic group (mdbook.GROUPS)

Kinds and their cleaning:
    mdbook   Markdown books (§2.1): hard-wrapped prose unwrapped, table
             padding collapsed, images and link targets dropped, U+FFFD
             removed. Books of class "keep" only (by default).
    edu, cosmo, web, math
             parquet "text" column, as published.
    code     StarCoderData (preferred) and github-code-clean (fallback,
             permissive licenses only, e-mails and IPv4 addresses redacted
             as StarCoderData does): path line + blank line + file; generated,
             minified and data-like files dropped. Provenance (repository,
             path, license) in src/meta (The Stack terms). JSON and YAML
             files get kind "data", so the mixture weights them apart.
    tool     Glaive, Hermes and xLAM conversations in one rendering:
             role blocks <system> <user> <assistant> <tool_response>, tool
             schemas in <tools>, calls as <tool_call>{"name", "arguments"}
             </tool_call> (plan 003 D9). The role tags are a pretraining
             rendering only; post-training fixes the real turn markers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from intgrad.corpus.build_shards import sanitize

# ---------------------------------------------------------------- mdbook

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_SPACES = re.compile(r"[ \t]{2,}")
_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s")
_BLANKS = re.compile(r"\n{3,}")


def _is_wrapped(lines: list[str]) -> bool:
    """A hard-wrapped prose paragraph: several lines, all but the last close
    to a common width. Verse and lists have ragged or marked lines."""
    if len(lines) < 2:
        return False
    body = lines[:-1]
    if any(_LIST.match(x) or x.lstrip().startswith(("#", "|", ">", "$$")) for x in lines):
        return False
    w = max(len(x.rstrip()) for x in body)
    if w < 40:
        return False
    return sum(len(x.rstrip()) >= 0.75 * w for x in body) >= 0.7 * len(body)


def _join(lines: list[str]) -> str:
    out = lines[0].rstrip()
    for nxt in lines[1:]:
        nxt = nxt.strip()
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and out[-2].islower() \
                and nxt[:1].isalpha() and nxt[:1].islower():
            out = out[:-1] + nxt                    # hyphenation at a line break
        else:
            out = f"{out} {nxt}"
    return out


def clean_markdown(text: str) -> str:
    text = text.replace("�", "")
    out, block, fence = [], [], False

    def flush():
        if block:
            out.append(_join(block) if _is_wrapped(block) else "\n".join(x.rstrip() for x in block))
            block.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            flush()
            fence = not fence
            out.append(line.rstrip())
            continue
        if fence:
            out.append(line.rstrip())
            continue
        line = _IMG.sub("", line)
        line = _LINK.sub(r"\1", line)
        s = line.strip()
        if s.startswith("|"):
            flush()
            out.append("|---|" if _TABLE_SEP.match(s) else _SPACES.sub(" ", s))
            continue
        if not s:
            flush()
            out.append("")
            continue
        block.append(line)
    flush()
    return _BLANKS.sub("\n\n", "\n".join(out)).strip()


def _clean_book(args):
    path, rel = args
    raw = Path(path).read_bytes().decode("utf-8", "replace")
    return rel, sanitize(clean_markdown(raw).encode("utf-8"))


def iter_mdbooks(src: Path, scores: Path, classes: set[str], langs: set[str], jobs: int = 0):
    import multiprocessing as mp
    rows = [r for r in map(json.loads, scores.read_text().splitlines())
            if r["class"] in classes and r["lang"] in langs]
    info = {r["path"]: r for r in rows}
    with mp.Pool(jobs or None) as pool:
        for rel, doc in pool.imap(_clean_book, [(str(src / r["path"]), r["path"]) for r in rows],
                                  chunksize=4):
            r = info[rel]
            yield doc, f"mdbook:{rel}", "mdbook", r["lang"], {
                "lic": "proprietary", "gated": False, "private": True,
                "grp": f"mdbook:{rel}", "group": r["group"]}


# ---------------------------------------------------------------- parquet text

TEXT_KINDS = {"edu-en": ("edu", "en"), "cosmo": ("cosmo", "en"), "cosmo-math": ("cosmo", "en"),
              "cosmo-stories": ("cosmo", "en"), "web-de": ("web", "de"), "math": ("math", "en")}


def _parquet_rows(path: Path, columns: list[str]):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=columns).to_pylist()
        yield from t


def iter_text(path: Path, name: str, rec: dict):
    kind, lang = TEXT_KINDS[name]
    cols = ["text"] + (["url"] if kind in ("edu", "web", "math") else [])
    for i, row in enumerate(_parquet_rows(path, cols)):
        t = row["text"]
        if not t:
            continue
        key = row.get("url") or f"{rec['path']}#{i}"
        yield sanitize(t.encode("utf-8")), f"{name}:{rec['path']}#{i}", kind, lang, {
            "lic": rec["license"], "gated": rec.get("gated", False), "private": False,
            "grp": f"{kind}:{key}"}


# ---------------------------------------------------------------- code

PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0", "unlicense"}
_GENERATED = re.compile(r"auto-?generated|do not edit|generated by|code generated", re.I)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_LOCK = re.compile(r"(?:^|/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|"
                   r"poetry\.lock|composer\.lock|go\.sum)$")
DATA_LIKE = {"json": 8_000, "yaml": 8_000}             # max bytes for data-like formats


def code_ok(content: str, path: str, lang: str, max_bytes: int = 100_000) -> bool:
    n = len(content)
    if n < 64 or n > DATA_LIKE.get(lang, max_bytes) or _LOCK.search(path):
        return False
    lines = content.split("\n")
    if max(map(len, lines)) > 1000 or n / len(lines) > 120:
        return False                                   # minified or data dump
    if sum(c.isalnum() for c in content) < 0.25 * n:
        return False
    return not _GENERATED.search(content[:2000])


def redact(s: str) -> str:
    return _IPV4.sub("<IP_ADDRESS>", _EMAIL.sub("<EMAIL>", s))


def iter_code(path: Path, name: str, rec: dict):
    scd = name.startswith("scd-")
    lang = name[4:] if scd else None
    if scd:
        cols = ["content", "max_stars_repo_name", "max_stars_repo_path"]
        if name == "scd-jupyter":
            cols = ["content", "repo_name", "path", "license"]
    else:
        cols = ["code", "repo_name", "path", "language", "license"]
    for row in _parquet_rows(path, cols):
        if scd:
            text = row["content"]
            repo = row.get("max_stars_repo_name") or row.get("repo_name") or ""
            fpath = row.get("max_stars_repo_path") or row.get("path") or ""
            lic = row.get("license") or "permissive (StarCoderData)"
        else:
            if (row["license"] or "").lower() not in PERMISSIVE:
                continue
            text, repo, fpath, lic = redact(row["code"]), row["repo_name"], row["path"], row["license"]
            lang = (row["language"] or "").lower()
        if not text or not code_ok(text, fpath, lang or ""):
            continue
        doc = sanitize(f"{fpath}\n\n{text}".encode("utf-8"))
        kind = "data" if lang in DATA_LIKE else "code"
        yield doc, f"{name}:{repo}:{fpath}", kind, "zxx", {
            "lic": lic, "gated": rec.get("gated", False), "private": False,
            "grp": f"code:{repo}", "repo": repo, "path": fpath}


# ---------------------------------------------------------------- tool use

def _block(tag: str, body: str) -> str:
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def _call(name: str, args) -> str:
    return _block("tool_call", json.dumps({"name": name, "arguments": args}, ensure_ascii=False))


def _schemas(objs: list) -> str:
    return _block("tools", "\n".join(json.dumps(o, ensure_ascii=False) for o in objs))


def _json_objects(s: str) -> list:
    """All top-level JSON objects in s, in order (Glaive's system prompt)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (j := s.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(s, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        out.append(obj)
        i = end
    return out


_GLAIVE_TURN = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")


def render_glaive(system: str, chat: str) -> str | None:
    head = system.removeprefix("SYSTEM:").strip()
    tools = _json_objects(head)
    cut = head.find("{")
    intro = (head[:cut] if cut >= 0 else head).strip().rstrip("-").strip()
    sys_body = intro + ("\n" + _schemas(tools) if tools else "")
    parts = [_block("system", sys_body)] if sys_body else []
    bits = _GLAIVE_TURN.split(chat.replace("<|endoftext|>", ""))
    for role, text in zip(bits[1::2], bits[2::2]):
        text = text.strip()
        if role == "USER":
            parts.append(_block("user", text))
        elif role == "FUNCTION RESPONSE":
            parts.append(_block("tool_response", text))
        else:
            m = re.search(r"<functioncall>\s*(\{.*\})\s*$", text, re.S)
            if m:
                raw = m.group(1)
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError:
                    try:                                # arguments given as '...' string
                        call = json.loads(re.sub(r"'(\{.*\})'", r"\1", raw, flags=re.S))
                    except json.JSONDecodeError:
                        return None
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return None
                pre = text[:m.start()].strip()
                text = (pre + "\n" if pre else "") + _call(call.get("name", ""), args)
            parts.append(_block("assistant", text))
    return "\n".join(parts) if len(parts) > 1 else None


_HERMES_ROLE = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool_response"}


def render_hermes(conv: list[dict]) -> str | None:
    parts = []
    for c in conv:
        role = _HERMES_ROLE.get(c.get("from"))
        v = (c.get("value") or "").strip()
        if not role or not v:
            continue
        if role == "tool_response" and v.startswith("<tool_response>"):
            parts.append(v)                              # already wrapped
        else:
            parts.append(_block(role, v))
    return "\n".join(parts) if len(parts) > 1 else None


def render_xlam(query: str, tools: str, answers: str) -> str | None:
    try:
        t, a = json.loads(tools), json.loads(answers)
    except json.JSONDecodeError:
        return None
    calls = "\n".join(_call(x.get("name", ""), x.get("arguments", {})) for x in a)
    return "\n".join([_block("system", _schemas(t)), _block("user", query), _block("assistant", calls)])


def iter_tool(path: Path, name: str, rec: dict, stats: dict):
    data = json.loads(path.read_text())
    meta = {"lic": rec["license"], "gated": rec.get("gated", False), "private": False}
    for i, x in enumerate(data):
        if name == "tool-glaive":
            doc, cid = render_glaive(x.get("system", ""), x.get("chat", "")), f"glaive:{i}"
        elif name == "tool-hermes":
            doc, cid = render_hermes(x.get("conversations", [])), f"hermes:{x.get('id', i)}"
        else:
            doc, cid = render_xlam(x["query"], x["tools"], x["answers"]), f"xlam:{x['id']}"
        if doc is None:
            stats["tool_unparsed"] = stats.get("tool_unparsed", 0) + 1
            continue
        yield sanitize(doc.encode("utf-8")), f"{name}:{path.name}#{i}", "tool", "en", {**meta, "grp": cid}


# ---------------------------------------------------------------- registry

def iter_hf(root: Path, names: set[str], exclude_gated: bool, stats: dict):
    """Every downloaded file of the selected ig-fetch sources, in registry order."""
    for rec in map(json.loads, (root / "SOURCES.jsonl").read_text().splitlines()):
        if rec["name"] not in names or (exclude_gated and rec.get("gated")):
            continue
        p = root / rec["repo"] / rec["path"]
        print(f"  {rec['name']}: {rec['path']}", file=sys.stderr, flush=True)
        if rec["name"] in TEXT_KINDS:
            yield from iter_text(p, rec["name"], rec)
        elif rec["name"] == "code" or rec["name"].startswith("scd-"):
            yield from iter_code(p, rec["name"], rec)
        elif rec["name"].startswith("tool-"):
            yield from iter_tool(p, rec["name"], rec, stats)
