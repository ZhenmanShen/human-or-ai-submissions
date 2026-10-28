"""Validate and normalise an extra labeled dataset before using it for training.

Takes a CSV (or a folder of .txt files) and writes data/external/<name>.csv with
columns text,label,source. It also:
  * drops texts outside the competition's length range (75-350 words by default)
  * optionally splits long texts into ~210-word chunks (--chunk)
  * drops exact duplicates and anything that also appears in train/test/unlabeled

  python scripts/add_external_data.py --input my_poems.csv --label 0 --name human_poems
  python scripts/add_external_data.py --input folder_of_txt/ --label 0 --name gutenberg --chunk
  python scripts/add_external_data.py --input gen.csv --name generated_qwen   # file already has label column

Then train with it:
  python train.py --model tfidf_logreg --set "data.external_files=[data/external/human_poems.csv]"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def chunk_words(text: str, target: int, min_w: int) -> list[str]:
    words = text.split()
    chunks = [" ".join(words[i:i + target]) for i in range(0, len(words), target)]
    return [c for c in chunks if len(c.split()) >= min_w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", type=int, choices=[0, 1], help="label for every row (if the CSV has none)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--chunk", action="store_true", help="split long texts into ~210-word chunks")
    ap.add_argument("--min-words", type=int, default=75)
    ap.add_argument("--max-words", type=int, default=350)
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

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="processing"):
        texts = chunk_words(str(r.text), 210, args.min_words) if args.chunk else [str(r.text).strip()]
        rows += [{"text": t, "label": int(r.label), "source": r.source} for t in texts]
    out = pd.DataFrame(rows)
    n = out["text"].str.split().str.len()
    out = out[(n >= args.min_words) & (n <= args.max_words)].drop_duplicates("text")

    existing = set()
    for f in ["train.csv", "test.csv", "unlabeled.csv"]:
        p = ROOT / "data" / f
        if p.exists():
            existing |= set(pd.read_csv(p)["text"])
    before = len(out)
    out = out[~out["text"].isin(existing)]

    dest = ROOT / "data" / "external" / f"{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} rows to {dest} (removed {before - len(out)} overlapping with competition data)")
    print(out["label"].value_counts().rename({0: "human", 1: "AI"}).to_string())


if __name__ == "__main__":
    main()
