"""Build the held-out evaluation sets (data/eval/) from generators that are never trained on.

These sets are for EVALUATION ONLY (config `data.eval_files`) -- never put them in
`data.external_files`. They estimate what train CV cannot: how a model does on AI text
from unseen generators, and on the creative writing that dominates test.

  python scripts/build_eval_set.py generate --generators meta-llama/Llama-3.2-3B-Instruct --device cuda:0
  python scripts/build_eval_set.py assemble

generate: for each generator (several decoding settings mixed per passage, --sample-mix):
  newgen    rewrites of 120 random train.csv HUMAN passages (base models: continuations of their
            opening) -> AI text on train's domains. Each row keeps `seed_id`, the train id it came
            from, so scripts/eval_ood.py can score it with the CV fold model that never saw that seed.
  poem      40 poems on titles of held-out Gutenberg poems (data/external/raw_gutenberg_poetry_eval.csv)
  poem_rw   40 poems written "in the style of" a held-out Gutenberg poem
  story     40 stories starting with the first 10 words of a held-out Gutenberg story
  Raw output goes to data/eval/raw/, then scripts/add_external_data.py cleans it (75-350 words,
  truncation at a line/sentence end, overlap filter against train/test/unlabeled) into data/eval/parts/.
assemble: data/eval/eval_newgen.csv  (AI only: text,label,generator,mode,seed_id)
          data/eval/eval_creative.csv (held-out Gutenberg human poems/stories, label 0, + the AI
                                       poem/poem_rw/story rows, label 1)
The Gutenberg books are split 80/20 by book (fetch_gutenberg.py); only the 20% "eval" books are used
here. Their human passages are archaic, the AI ones modern, so eval_creative overstates separability
-- compare models on it, don't read its AUROC as a test-score estimate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
GUT = "data/external/raw_gutenberg_{}_eval.csv"
JOBS = {  # part -> generate_ai_text.py arguments
    "newgen": ["--source", "data/train.csv", "--mode", "rewrite", "--n", "120"],
    "poem": ["--source", GUT.format("poetry"), "--title-col", "title", "--mode", "poem", "--n", "40"],
    "poem_rw": ["--source", GUT.format("poetry"), "--mode", "rewrite", "--n", "40"],
    "story": ["--source", GUT.format("stories"), "--mode", "story", "--prefix-words", "10", "--n", "40"],
}
HUMAN_CREATIVE = ["data/external/human_poetry_eval.csv", "data/external/human_stories_eval.csv"]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def generate(args):
    for gen in args.generators:
        short = gen.split("/")[-1]
        for part, extra in JOBS.items():
            out = EVAL / "parts" / f"{short}__{part}.csv"
            if out.exists() and not args.force:
                print(f"skip {out.name} (exists)")
                continue
            seed = args.seed + zlib.crc32(gen.encode()) % 10000  # different seed passages per generator
            raw = EVAL / "raw"
            run([sys.executable, "scripts/generate_ai_text.py", "--model", gen, *extra, "--sample-mix",
                 "--seed", str(seed), "--tag", part, "--batch-size", str(args.batch_size),
                 "--device", args.device, "--out-dir", str(raw.relative_to(ROOT))])
            mode = extra[extra.index("--mode") + 1]
            raw_file = raw / f"raw_gen_{short}_{mode}_{part}_s{seed}.csv"
            run([sys.executable, "scripts/add_external_data.py", "--input", str(raw_file), "--name",
                 f"{short}__{part}", "--truncate", "--out-dir", str(out.parent.relative_to(ROOT))])


def assemble(_args):
    parts = []
    for f in sorted((EVAL / "parts").glob("*__*.csv")):
        gen, part = f.stem.split("__")
        d = pd.read_csv(f)
        d["generator"], d["mode"] = gen, part
        parts.append(d)
    ai = pd.concat(parts, ignore_index=True)
    cols = ["text", "label", "generator", "mode", "seed_id"]
    newgen = ai[ai["mode"] == "newgen"][cols]
    human = pd.concat([pd.read_csv(ROOT / f).assign(generator="human", mode=Path(f).stem.split("_")[1])
                       for f in HUMAN_CREATIVE], ignore_index=True)
    creative = pd.concat([human, ai[ai["mode"] != "newgen"]], ignore_index=True)[cols[:-1]]
    for name, d in [("eval_newgen", newgen), ("eval_creative", creative)]:
        d = d.reset_index(drop=True)
        d.insert(0, "id", [f"{name}_{i}" for i in range(len(d))])
        d.to_csv(EVAL / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} rows")
        print(d.groupby(["generator", "mode"]).size().unstack(fill_value=0).to_string(), "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--generators", nargs="+", required=True)
    g.add_argument("--device", default="auto")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seed", type=int, default=1000)
    g.add_argument("--force", action="store_true", help="regenerate parts that already exist")
    sub.add_parser("assemble")
    args = ap.parse_args()
    (EVAL / "parts").mkdir(parents=True, exist_ok=True)
    generate(args) if args.cmd == "generate" else assemble(args)


if __name__ == "__main__":
    main()
