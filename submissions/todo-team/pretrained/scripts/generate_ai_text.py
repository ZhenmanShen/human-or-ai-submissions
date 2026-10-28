"""Generate extra AI-written passages with a local Hugging Face model.

Strategy: take human-written passages (train.csv label 0, or any CSV with a text
column, e.g. Gutenberg poems from scripts/fetch_gutenberg.py) and ask the model
to continue / rewrite them, or to write a poem / story / plot summary on the
same title, so the new AI texts cover the same topics as the human ones.
Different models, prompts and decoding settings (temperature, top_p,
repetition_penalty, greedy) produce different "AI styles" -- variety is the
point, since the test set contains generators not seen in training.

Instruct models get a chat prompt; base models (no chat template, e.g. gpt2-large,
opt-1.3b) get a plain prefix to continue. Generations are batched.

  python scripts/generate_ai_text.py --model Qwen/Qwen2.5-0.5B-Instruct --n 300
  python scripts/generate_ai_text.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --mode poem \
      --source data/external/raw_gutenberg_poetry.csv --n 400 --sample-mix
  python scripts/add_external_data.py --input data/external/raw_gen_<...>.csv --name gen_x --truncate
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Several phrasings per mode, picked at random per row (prompt variety -> style variety).
PROMPTS = {
    "continue": ["Continue the following text in the same style and format. Write about {words} words.\n\n{prefix}"],
    "rewrite": ["Write a new piece of about {words} words on the same topic and in the same format as this example. "
                "Only output the text itself.\n\nExample:\n{full}"],
    "poem": ["Write a poem titled \"{title}\".",
             "Write a poem of about {words} words that begins with the line: \"{first_line}\"",
             "Compose a poem inspired by the phrase \"{title}\".",
             "Write the body of a poem titled \"{title}\". Do not include the title.",
             "Write a free-verse poem about {title_lc}."],
    "story": ["Write a short story of about {words} words that starts like this: \"{prefix}\"",
              "Write a short story inspired by: {title}",
              "Write the opening of a novel. Begin with: \"{prefix}\"",
              "Write a {words}-word piece of fiction. Start your story with the words \"{prefix}\""],
    "plot": ["Write a plot summary of about {words} words for a novel titled \"{title}\".",
             "Summarize the plot of the book \"{title}\" in about {words} words.",
             "Write the body of a Wikipedia-style plot summary for a novel called \"{title}\"."],
}
# What base (non-chat) models are asked to continue.
BASE_PROMPTS = {
    "continue": "{prefix}", "rewrite": "{prefix}", "story": "{prefix}",
    "poem": "{title}\n\n", "plot": "{title}\n\nPlot summary\n\n",
}


def fields(text: str, title: str | None, prefix_words: int) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    first = lines[0] if lines else text[:60]
    t = (title or " ".join(first.split()[:8])).strip().strip('"').rstrip(",;:")
    return {"words": int(np.clip(len(text.split()), 120, 300)), "prefix": " ".join(text.split()[:prefix_words]),
            "full": text, "title": t, "title_lc": t[0].lower() + t[1:] if t else t, "first_line": first}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--source", default="data/train.csv", help="CSV with text (and optional label / title) column")
    ap.add_argument("--title-col", default=None, help="column with titles for poem/story/plot (default: first line)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=list(PROMPTS), default="continue")
    ap.add_argument("--prefix-words", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=420)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--sample-mix", action="store_true",
                    help="per row: random temperature 0.6-1.2, top_p 0.9-1.0, 15%% greedy (overrides the above)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="extra tag for the output file name")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:<i> | mps | cpu")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils import device, hf_source, seed_everything

    seed_everything(args.seed)
    dev = device(args.device)
    print(f"Generating on device: {dev}")
    src = hf_source(args.model)
    if dev.startswith("cuda"):
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dt = torch.float16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(src)
    tok.padding_side = "left"  # decoder-only batch generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dt)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dt)
    model = model.to(dev).eval()
    chat = bool(tok.chat_template)

    data = pd.read_csv(ROOT / args.source)
    if "label" in data:
        data = data[data["label"] == 0]
    data = data.sample(min(args.n, len(data)), random_state=args.seed).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    jobs = []
    for i, r in data.iterrows():
        f = fields(str(r["text"]), r[args.title_col] if args.title_col else None, args.prefix_words)
        if chat:
            tmpl = PROMPTS[args.mode][rng.integers(len(PROMPTS[args.mode]))]
            prompt = tok.apply_chat_template([{"role": "user", "content": tmpl.format(**f)}],
                                             tokenize=False, add_generation_prompt=True)
        else:
            tmpl = BASE_PROMPTS[args.mode]
            prompt = tmpl.format(**f)
        if args.sample_mix:
            greedy = rng.random() < 0.15
            dec = {"do_sample": not greedy, "temperature": float(rng.uniform(0.6, 1.2)),
                   "top_p": float(rng.uniform(0.9, 1.0)), "repetition_penalty": float(rng.choice([1.0, 1.1, 1.2]))}
        else:
            dec = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p,
                   "repetition_penalty": args.repetition_penalty}
        jobs.append((prompt, dec, f, tmpl))

    # Rows with identical decoding settings can share a batch; with --sample-mix that's per row,
    # so group by rounded settings to keep batches reasonably full.
    def key(d):
        return (d["do_sample"], round(d["temperature"], 1), round(d["top_p"], 2), d["repetition_penalty"])
    order = sorted(range(len(jobs)), key=lambda i: key(jobs[i][1]))
    rows = [None] * len(jobs)
    bar = tqdm(total=len(jobs), desc=f"generating ({args.model})")
    i = 0
    while i < len(order):
        k = key(jobs[order[i]][1])
        idx = [order[i]]
        while i + len(idx) < len(order) and len(idx) < args.batch_size and key(jobs[order[i + len(idx)]][1]) == k:
            idx.append(order[i + len(idx)])
        i += len(idx)
        d = dict(jobs[idx[0]][1])
        d["temperature"], d["top_p"] = k[1], k[2]
        if not d["do_sample"]:
            d.pop("temperature"), d.pop("top_p")
        enc = tok([jobs[j][0] for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=not chat).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, **d)
        for b, j in enumerate(idx):
            gen = tok.decode(out[b, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            prompt_text, _, f, tmpl = jobs[j]
            if args.mode == "continue" or (not chat and args.mode == "story"):
                gen = f["prefix"] + " " + gen
            elif not chat and args.mode in ("poem", "plot"):
                gen = prompt_text.lstrip() + gen
            rows[j] = {"text": gen, "label": 1,
                       "source": f"{args.model}|{args.mode}|{'chat' if chat else 'base'}|"
                                 f"{'greedy' if not d['do_sample'] else 'T=%.1f' % k[1]}|rp={k[3]}"}
        bar.update(len(idx))
    bar.close()

    name = args.model.split("/")[-1]
    dest = ROOT / "data" / "external" / f"raw_gen_{name}_{args.mode}{'_' + args.tag if args.tag else ''}_s{args.seed}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"Wrote {len(rows)} generated passages to {dest}")
    print("Next: python scripts/add_external_data.py --input", dest.relative_to(ROOT), "--name <name> --truncate")


if __name__ == "__main__":
    main()
