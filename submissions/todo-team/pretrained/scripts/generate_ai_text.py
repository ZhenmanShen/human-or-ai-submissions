"""Generate extra AI-written passages with a local Hugging Face model.

Strategy: take the opening of human-written passages (from train.csv label 0,
or any CSV with a text column), and ask the model to continue / rewrite them,
so the new AI texts cover the same topics and formats as the human ones.
Different models, prompts and decoding settings (temperature, top_p,
repetition_penalty) produce different "AI styles" -- variety is the point,
since the test set contains generators not seen in training.

Runs on the M1 Pro via MPS or on an NVIDIA GPU (--device cuda). Small instruct models (0.5B-1.5B) are practical
on 16 GB; expect roughly a few seconds per passage.

  python scripts/generate_ai_text.py --model Qwen/Qwen2.5-0.5B-Instruct --n 300
  python scripts/generate_ai_text.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --n 300 --temperature 1.0 --mode continue
  python scripts/add_external_data.py --input data/external/raw_gen_<...>.csv --name gen_qwen05
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":  # LightGBM + PyTorch each ship libomp on macOS; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = {
    "continue": "Continue the following text in the same style and format. Write about {words} words.\n\n{prefix}",
    "rewrite": "Write a new piece of about {words} words on the same topic and in the same format as this example. "
               "Only output the text itself.\n\nExample:\n{full}",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--source", default="data/train.csv", help="CSV with text (and optional label) column")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=list(PROMPTS), default="continue")
    ap.add_argument("--prefix-words", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=380)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:<i> | mps | cpu")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils import device, seed_everything

    seed_everything(args.seed)
    dev = device(args.device)
    print(f"Generating on device: {dev}")
    from src.utils import hf_source
    src = hf_source(args.model)
    dt = torch.float16 if dev != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(src)
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dt)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dt)
    model = model.to(dev).eval()

    src = pd.read_csv(ROOT / args.source)
    if "label" in src:
        src = src[src["label"] == 0]
    src = src.sample(min(args.n, len(src)), random_state=args.seed)

    rows = []
    for text in tqdm(src["text"], desc=f"generating ({args.model})"):
        words = len(text.split())
        prefix = " ".join(text.split()[: args.prefix_words])
        prompt = PROMPTS[args.mode].format(words=words, prefix=prefix, full=text)
        if tok.chat_template:
            prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if args.mode == "continue":
            gen = prefix + " " + gen
        rows.append({"text": gen, "label": 1,
                     "source": f"{args.model}|{args.mode}|T={args.temperature}"})

    tag = args.model.split("/")[-1]
    dest = ROOT / "data" / "external" / f"raw_gen_{tag}_{args.mode}_s{args.seed}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"Wrote {len(rows)} generated passages to {dest}")
    print("Next: python scripts/add_external_data.py --input", dest.relative_to(ROOT), "--name <name>")


if __name__ == "__main__":
    main()
