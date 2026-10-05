"""Score the framing-matched corpus docs with a second sentiment model (RoBERTa).

The examination board asked for a second model to compare against VADER, with an
agreement rate between the two. We score only the documents that match one of the
three metric-framing regexes, since that is the population the framing finding
rests on; analyze_corpus.py picks the output up (if present) and prints the
agreement numbers.

Usage:
    python scripts/score_roberta.py            # all framing-matched docs (~34k)
    python scripts/score_roberta.py --limit 100  # smoke test

Crash-safe: appends one fsync'd JSONL line per doc to data/roberta_sentiment.jsonl;
re-run the same command to resume (skips done keys, trims any partial last line).
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUT_DEFAULT = "data/roberta_sentiment.jsonl"

# keep in sync with RX in analyze_corpus.py
RX = {
    "dora": re.compile(r"\bdora\b|deployment frequency|change failure|"
                       r"mean time to recover|four key", re.I),
    "individual": re.compile(r"\bpip\b|performance improvement plan|performance review|"
                             r"\bstory points?\b|\bvelocity\b|\bokrs?\b|\bkpis?\b|"
                             r"lines of code", re.I),
    "scrum": re.compile(r"\bscrum\b|\bagile\b", re.I),
}


def load_done_keys(path):
    """Read existing JSONL, return set of done keys, and rewrite the file keeping only
    valid lines (drops any trailing partial line from a mid-write crash)."""
    if not os.path.exists(path):
        return set()
    done, valid = set(), []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial/corrupt (only possible on the last line)
            if "key" in obj:
                done.add(obj["key"])
                valid.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        if valid:
            fh.write("\n".join(valid) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--corpus", default=None, help="corpus CSV (default: newest data/*_corpus.csv)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="process at most N docs (0 = all)")
    args = ap.parse_args()

    corpus = args.corpus or max(glob.glob("data/*_corpus.csv"), key=os.path.getctime)
    df = pd.read_csv(corpus, dtype={"title": str, "selftext": str}, low_memory=False)
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    # same text VADER scores in analyze_corpus.py (RoBERTa is case-aware, but the
    # comparison must be on identical input)
    df["text"] = (df["title"] + " " + df["selftext"]).str.lower()

    mask = pd.Series(False, index=df.index)
    for rx in RX.values():
        mask |= df["text"].str.contains(rx)
    docs = df[mask].copy()
    # post and comment id spaces can collide, so key on both
    docs["key"] = docs["type"] + ":" + docs["id"].astype(str)
    print(f"corpus: {corpus}  | framing-matched docs: {len(docs):,}")

    done = load_done_keys(args.out)
    todo = docs[~docs["key"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"already done: {len(done):,}  | to process now: {len(todo):,}")
    if todo.empty:
        print("nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    if device == "cuda":
        model = model.half()
    model = model.to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"model: {MODEL}  | device: {device}  | labels: {labels}")

    keys = todo["key"].tolist()
    texts = todo["text"].tolist()
    with open(args.out, "a", encoding="utf-8") as fh:
        for start in tqdm(range(0, len(texts), args.batch), unit="batch"):
            batch_keys = keys[start:start + args.batch]
            enc = tokenizer(texts[start:start + args.batch], max_length=256,
                            padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
            for key, p in zip(batch_keys, probs):
                p = {lab: round(float(v), 4) for lab, v in zip(labels, p)}
                fh.write(json.dumps({"key": key,
                                     "label": max(p, key=p.get),
                                     "p_neg": p["negative"],
                                     "p_neu": p["neutral"],
                                     "p_pos": p["positive"]}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
