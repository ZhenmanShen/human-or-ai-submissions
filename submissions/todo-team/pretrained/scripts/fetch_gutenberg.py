"""Fetch human-written passages (poetry or prose fiction) from Project Gutenberg.

Why: the test set is ~15% poems and ~28% stories / narrative text, versus <1% and ~8% of
train (see AGENTS.md, "topic shift"). Public-domain books are a human-text source that is
not an AI-detection benchmark. Passages are shaped like the competition's human texts:
75-300 words, poem line breaks kept, prose hard-wraps undone, and no blank lines (human
passages in train.csv never contain "\\n\\n"; stanza/paragraph breaks become "\\n").

  python scripts/fetch_gutenberg.py --kind poetry --n-books 120 --name gutenberg_poetry
  python scripts/fetch_gutenberg.py --kind prose --subject "Short stories" --n-books 120 --name gutenberg_stories
  python scripts/add_external_data.py --input data/external/raw_gutenberg_poetry.csv --label 0 --name human_poetry

Writes data/external/raw_<name>.csv (text, title, source). Run it through
scripts/add_external_data.py, which removes anything overlapping the competition data.
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
JUNK = re.compile(r"^\s*(\[.*\]|CHAPTER\b.*|BOOK\b.*|CANTO\b.*|PART\b.*|[IVXLC]+\.?|\d+\.?|_+|\*+(\s*\*)*|"
                  r"(Illustration|Footnote|Transcriber).*|Contents|CONTENTS|INDEX|NOTES?)\s*$")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "cs5785-course-project (research; low volume)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace").replace("\r\n", "\n")


def body(text: str) -> str:
    """Book text between the Project Gutenberg START/END markers."""
    text = text.replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", text)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    return text[s.end() if s else 0:e.start() if e else len(text)]


def blocks(text: str) -> list[list[str]]:
    """Blank-line separated blocks (stanzas / paragraphs) as lists of stripped lines."""
    out = []
    for b in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in b.split("\n") if l.strip() and not JUNK.match(l)]
        if lines:
            out.append(lines)
    return out


def is_verse(lines: list[str]) -> bool:
    return len(lines) >= 2 and np.mean([len(l) for l in lines]) < 60 and max(len(l) for l in lines) < 90


def passages(text: str, kind: str, target: int, max_words: int) -> list[str]:
    """Group consecutive stanzas / paragraphs into passages of about `target` words."""
    out, cur, n = [], [], 0
    for lines in blocks(body(text)):
        if kind == "poetry":
            if not is_verse(lines):  # prose preface, notes, headings -> ends the current passage
                if n >= 75:
                    out.append("\n".join(cur))
                cur, n = [], 0
                continue
            unit = "\n".join(lines)
        else:
            if is_verse(lines) or sum(len(l.split()) for l in lines) < 8:  # headings, verse, dialogue fragments
                continue
            unit = " ".join(lines)  # undo Gutenberg's 70-column hard wrap
        w = len(unit.split())
        if n and n + w > max_words:
            if n >= 75:
                out.append("\n".join(cur))
            cur, n = [], 0
        if w > max_words:
            continue
        cur.append(unit)
        n += w
        if n >= target:
            out.append("\n".join(cur))
            cur, n = [], 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["poetry", "prose"], required=True)
    ap.add_argument("--subject", help="regex on the catalog's Subjects (default: Poetry / Fiction)")
    ap.add_argument("--n-books", type=int, default=100)
    ap.add_argument("--per-book", type=int, default=8, help="max passages kept per book (variety)")
    ap.add_argument("--target-words", type=int, default=0, help="0 = random 120-260 per passage, like the data")
    ap.add_argument("--max-words", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between downloads (be polite)")
    args = ap.parse_args()

    cache = ROOT / "data" / "external" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cat_path = cache / "pg_catalog.csv"
    if not cat_path.exists():
        cat_path.write_text(fetch(CATALOG_URL), encoding="utf-8", newline="\n")  # no \r\n doubling on Windows
    cat = pd.read_csv(cat_path)
    cat = cat[(cat["Language"] == "en") & (cat["Type"] == "Text")]
    subject = args.subject or ("Poetry" if args.kind == "poetry" else "Fiction")
    cat = cat[cat["Subjects"].fillna("").str.contains(subject, case=False, regex=True)]
    if args.kind == "prose":
        cat = cat[~cat["Subjects"].fillna("").str.contains("Poetry", case=False)]
    books = cat.sample(min(args.n_books, len(cat)), random_state=args.seed)
    print(f"{len(cat)} catalog books match '{subject}'; fetching {len(books)}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, b in tqdm(books.iterrows(), total=len(books), desc="books"):
        path = cache / f"pg{b['Text#']}.txt"
        try:
            if not path.exists():
                path.write_text(fetch(BOOK_URL.format(id=b["Text#"])), encoding="utf-8", newline="\n")
                time.sleep(args.delay)
        except OSError as e:  # missing file / network hiccup: skip this book, report it
            tqdm.write(f"  skip {b['Text#']}: {e}")
            continue
        target = args.target_words or int(rng.integers(120, 261))
        ps = passages(path.read_text(encoding="utf-8"), args.kind, target, args.max_words)
        if len(ps) > args.per_book:
            ps = [ps[i] for i in sorted(rng.choice(len(ps), args.per_book, replace=False))]
        rows += [{"text": p, "title": str(b["Title"]).split("\n")[0], "source": f"gutenberg:{b['Text#']}"}
                 for p in ps]
    out = pd.DataFrame(rows)
    dest = ROOT / "data" / "external" / f"raw_{args.name}.csv"
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} passages from {out['source'].nunique() if len(out) else 0} books to {dest}")


if __name__ == "__main__":
    main()
