"""Validate and normalise an extra labeled dataset before using it for training.

Takes a CSV (or a folder of .txt files) and writes data/external/<name>.csv with
columns text,label,source. It also:
  * drops texts outside the competition's length range (75-350 words by default);
    --truncate instead cuts long texts at the last line / sentence end that fits
  * optionally splits long texts into ~210-word chunks (--chunk), keeping line breaks
  * --collapse-blank-lines: "\\n\\n+" -> "\\n". Human passages in train.csv never contain a
    blank line (66% of AI ones do), so extra *human* text must be formatted the same way
    or the model learns the opposite of what holds in the competition data.
  * drops exact duplicates, and anything sharing >= --max-shared 8-word shingles with
    train/test/unlabeled -- so a public-domain source can't smuggle test passages (or
    near copies of them) into training.

  python scripts/add_external_data.py --input my_poems.csv --label 0 --name human_poems --collapse-blank-lines
  python scripts/add_external_data.py --input folder_of_txt/ --label 0 --name gutenberg --chunk
  python scripts/add_external_data.py --input gen.csv --name generated_qwen --truncate  # file has a label column

Then train with it:
  python train.py --model tfidf_logreg --set "data.external_files=[data/external/human_poems.csv]"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHINGLE = 8


def chunk_words(text: str, target: int, min_w: int) -> list[str]:
    """Split into ~target-word chunks at line boundaries (poem lines / paragraphs stay intact)."""
    chunks, cur, n = [], [], 0
    for line in text.split("\n"):
        w = len(line.split())
        if n and n + w > target:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        chunks.append("\n".join(cur))
    return [c.strip() for c in chunks if len(c.split()) >= min_w]


def truncate(text: str, max_w: int) -> str:
    """Longest prefix of at most max_w words ending at a line break or sentence end."""
    if len(text.split()) <= max_w:
        return text
    words = list(re.finditer(r"\S+", text))
    head = text[:words[max_w - 1].end()]
    cut = max(head.rfind("\n"), *(head.rfind(p) + 1 for p in (". ", "! ", "? ", '." ', '!" ', '?" ')))
    return head[:cut].strip() if cut > 0 else head.strip()


def shingles(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", type=int, choices=[0, 1], help="label for every row (if the CSV has none)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--chunk", action="store_true", help="split long texts into ~210-word chunks")
    ap.add_argument("--truncate", action="store_true", help="cut texts longer than --max-words instead of dropping")
    ap.add_argument("--collapse-blank-lines", action="store_true", help="format like the competition's human texts")
    ap.add_argument("--min-words", type=int, default=75)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--max-shared", type=int, default=2,
                    help=f"drop rows sharing at least this many {SHINGLE}-word shingles with the competition data")
    ap.add_argument("--out-dir", default="data/external",
                    help="data/external for training data (data.external_files), data/eval for eval-only sets")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        df = pd.DataFrame({"text": [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(inp.glob("*.txt"))]})
    else:
        df = pd.read_csv(inp).rename(columns={args.text_col: "text"})
    if args.label is not None:
        df["label"] = args.label
    if "label" not in df:
        sys.exit("No label column: pass --label 0 (human) or --label 1 (AI)")
    df["source"] = df.get("source", args.name)
    df["text"] = df["text"].fillna("").astype(str).str.replace("\r\n", "\n")

    keep_cols = [c for c in ("seed_id", "title") if c in df]  # provenance, kept for eval-set analysis
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="processing"):
        text = re.sub(r"\n\s*\n+", "\n", r.text) if args.collapse_blank_lines else r.text
        texts = chunk_words(text, 210, args.min_words) if args.chunk else [text.strip()]
        if args.truncate:
            texts = [truncate(t, args.max_words) for t in texts]
        extra = {c: r[c] for c in keep_cols}
        rows += [{"text": t, "label": int(r.label), "source": r.source, **extra} for t in texts]
    out = pd.DataFrame(rows)
    n = out["text"].str.split().str.len()
    out = out[(n >= args.min_words) & (n <= args.max_words)].drop_duplicates("text")
    print(f"{len(out)} rows within {args.min_words}-{args.max_words} words")

    comp = pd.concat([pd.read_csv(ROOT / "data" / f)["text"] for f in ("train.csv", "test.csv", "unlabeled.csv")
                      if (ROOT / "data" / f).exists()])
    exact = set(comp)
    seen = set()
    for t in tqdm(comp, desc="competition shingles"):
        seen |= shingles(t)
    shared = out["text"].map(lambda t: len(shingles(t) & seen))
    keep = ~out["text"].isin(exact) & (shared < args.max_shared)
    print(f"removed {(~keep).sum()} rows overlapping the competition data "
          f"({out['text'].isin(exact).sum()} exact, {(shared >= args.max_shared).sum()} sharing "
          f">={args.max_shared} {SHINGLE}-grams)")
    out = out[keep]

    dest = ROOT / args.out_dir / f"{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} rows to {dest}")
    print(out["label"].value_counts().rename({0: "human", 1: "AI"}).to_string())


if __name__ == "__main__":
    main()
