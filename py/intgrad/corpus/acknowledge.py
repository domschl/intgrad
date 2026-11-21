"""ig-acknowledge — write DATASETS.md: every data source, its license, the
obligations we follow, and the citation its authors ask for (plan 003).

    uv run --project py ig-acknowledge [--out DATASETS.md] [--cards data/raw/hf/cards]

Hugging Face sources come from the ig-fetch registry. Their dataset cards
are fetched once (cached under --cards) and the BibTeX blocks in them are
copied verbatim. Sources from before plan 003 and the private local
sources are listed by hand below; private sources are described, never
named.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from intgrad.corpus.fetch import HUB, ROOT, SOURCES, _get

# What each license obliges us to do (shown once per dataset).
DUTY = {
    "ODC-By 1.0": "Attribution (this file). The source is a Common Crawl derivative, "
                  "so the Common Crawl terms of use also apply.",
    "Apache-2.0": "Attribution and license notice (this file).",
    "CC-BY-4.0": "Attribution (this file), and changes are stated: the records are converted "
                 "to the plan-003 tool-call format (tag around a JSON body).",
    "permissive per file (The Stack terms of use)":
        "The Stack terms of use: the original license of every file is respected, "
        "including attribution; provenance (repository, path, license) is kept per "
        "document in the corpus manifest; the local copy is updated when the "
        "maintainers publish removals; the data is not re-hosted.",
    "per file (kept: permissive only)": "Only files under permissive licenses are kept "
                                       "(the license field per file); provenance is kept "
                                       "per document in the corpus manifest.",
}

# Citations for datasets whose card has none (the paper that introduced them).
EXTRA_BIB = {
    "bigcode/starcoderdata": [
        """@article{li2023starcoder,
  title={StarCoder: may the source be with you!},
  author={Li, Raymond and Ben Allal, Loubna and Zi, Yangtian and Muennighoff, Niklas and others},
  journal={arXiv preprint arXiv:2305.06161},
  year={2023}
}""",
        """@article{kocetkov2022stack,
  title={The Stack: 3 TB of permissively licensed source code},
  author={Kocetkov, Denis and Li, Raymond and Ben Allal, Loubna and others},
  journal={arXiv preprint arXiv:2211.15533},
  year={2022}
}"""],
}

# Sources from before plan 003 (corpus-v1-wiki, corpus-v2) and private ones.
OTHER = [
    ("Project Gutenberg", "https://www.gutenberg.org/",
     "Public domain in the USA (per book; the Project Gutenberg license and trademark "
     "apply to the distributed files).",
     "Books from a local mirror. The Project Gutenberg header and license boilerplate "
     "are stripped (intgrad.corpus.gutenberg), and the trademark is not used."),
    ("Wikipedia (German, dump 2015-10-02; English, 2015 dump)", "https://dumps.wikimedia.org/",
     "CC BY-SA 3.0 and GFDL (text by the Wikipedia contributors).",
     "Article text extracted from pages-articles dumps (wiki-en, wiki-de)."),
    ("Wikipedia lead paragraphs, 2017 (\"wiki2017-defs\")", "https://en.wikipedia.org/",
     "CC BY-SA 3.0 (derived from English Wikipedia).",
     "Title and lead-paragraph pairs (wiki-defs)."),
    ("Private local sources", None,
     "Proprietary or unknown licenses.",
     "A private collection of Markdown books, and a small set of local texts. They are "
     "used locally for research only, never committed, quoted or distributed (plan 003 "
     "D2). Corpus builds can exclude them; they are not named here."),
]


def card(repo: str, cache: Path) -> str:
    f = cache / repo / "README.md"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        with _get(f"{HUB}/datasets/{repo}/resolve/main/README.md") as r:
            f.write_bytes(r.read())
    return f.read_text(errors="replace")


def bibtex(text: str) -> list[str]:
    """BibTeX entries in a card: fenced blocks first, else bare @type{...} entries."""
    out = []
    for block in re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.S):
        if re.search(r"@\w+\s*\{", block):
            out.append(block.strip())
    if not out:
        for m in re.finditer(r"@\w+\s*\{", text):
            depth, i = 0, m.start()
            for j in range(m.end() - 1, len(text)):
                depth += text[j] == "{"
                depth -= text[j] == "}"
                if depth == 0:
                    out.append(text[i:j + 1].strip())
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "DATASETS.md")
    ap.add_argument("--cards", type=Path, default=ROOT / "data" / "raw" / "hf" / "cards")
    a = ap.parse_args()
    repos: dict[str, dict] = {}
    for name, s in SOURCES.items():
        r = repos.setdefault(s.repo, {"license": s.license, "gated": s.gated, "parts": []})
        r["parts"] += [f"`{p}`" + (f" (first {s.limit} files)" if s.limit else "") if p else "all files"
                       for p in s.paths]
    L = ["# Datasets and acknowledgments", "",
         "intgrad trains on the sources below. We thank their authors and maintainers.",
         "This file is generated by `ig-acknowledge` (py/intgrad/corpus/acknowledge.py)",
         "from the download registry and the dataset cards; do not edit by hand.", "",
         "No data is stored in this repository. Downloaded files live in `data/` and in",
         "a local archive, pinned by revision with license and sha256 per file",
         "(`data/raw/hf/SOURCES.jsonl`). `ig-fetch --open-only` re-downloads only the",
         "ungated sources, and corpus builds can exclude gated and private sources.", "",
         "## Downloaded datasets (Hugging Face)", ""]
    for repo, r in repos.items():
        L += [f"### {repo}", "",
              f"- Source: {HUB}/datasets/{repo}",
              f"- License: {r['license']}" + (" — gated: terms accepted by the user before download"
                                               if r["gated"] else ""),
              f"- Used: {', '.join(r['parts'])}",
              f"- Obligations: {DUTY.get(r['license'], 'Attribution (this file).')}"]
        try:
            bib = bibtex(card(repo, a.cards))
        except OSError as e:
            bib, L = [], L + [f"- Citation: card not available ({e.__class__.__name__})"]
        bib = bib or EXTRA_BIB.get(repo, [])
        if bib:
            L += ["- Citation, as requested by the authors:", ""]
            L += [f"```bibtex\n{b}\n```" for b in bib]
        elif not any(x.startswith("- Citation") for x in L[-6:]):
            L += ["- Citation: none given on the dataset card; please cite the dataset page."]
        L.append("")
    L += ["## Other sources", ""]
    for title, url, lic, use in OTHER:
        L += [f"### {title}", ""] + ([f"- Source: {url}"] if url else []) + \
             [f"- License: {lic}", f"- Used: {use}", ""]
    a.out.write_text("\n".join(L).rstrip() + "\n")
    print(f"wrote {a.out} ({len(repos)} Hugging Face datasets, {len(OTHER)} other sources)")


if __name__ == "__main__":
    main()
