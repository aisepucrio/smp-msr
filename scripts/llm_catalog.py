"""
LLM-based open extraction for the metric catalog. Each Reddit POST is passed through
a local Gemma 3 GGUF model (llama-cpp-python) with a strict prompt that extracts every
productivity/mental-health metric, framework, practice, or well-being indicator that the
post explicitly discusses. Output is schema-constrained JSON.

    python scripts/llm_catalog.py --model models/gemma-3-4b-it-Q4_K_M.gguf
    python scripts/llm_catalog.py --model ... --limit 200      # sample / smoke test
    python scripts/llm_catalog.py --model ... --metrics-only   # only metrics_productivity posts

Crash-safe: results are appended to data/llm_labels.jsonl one post at a time with
flush()+fsync(), so an abrupt power-off loses at most the in-flight post. Re-running the
same command resumes (skips post ids already done). temperature=0 for reproducibility.
"""
import argparse
import glob
import json
import os
import time

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

OUT_DEFAULT = "data/llm_labels.jsonl"

SYSTEM_PROMPT = """You are an expert software-engineering research analyst. Given a single Reddit post by a software developer, extract ONLY items from these two narrow categories that the post EXPLICITLY mentions:

(A) PRODUCTIVITY MEASUREMENT: a metric, framework, or management practice used to measure, track, or evaluate software work, team delivery, or developer performance. Examples: velocity, story points, DORA, deployment frequency, lead time, change failure rate, MTTR, lines of code, code coverage, cycle time, throughput, OKRs, KPIs, performance reviews, PIPs, stack ranking, 1:1s, technical debt, number of commits, pull request counts, uptime/SLAs.
(B) MENTAL-HEALTH / WELL-BEING indicator explicitly stated: burnout, anxiety, stress, depression, exhaustion, overwork, work-life balance, micromanagement, toxic environment.

STRICT RULES:
- Only items actually present in the text. Do NOT infer or invent. Empty list if none.
- Do NOT extract generic nouns or activities that are not measurement-or-wellbeing, such as: job interviews, leetcode, coding challenges, side/personal projects, salary or compensation, specific languages/tools/companies, "career", "team", "time", "workload", "experience", "productivity" used only as a generic word.
- "surface": exact phrase as written (lowercased). "canonical": standard name (e.g., "Velocity", "DORA", "Performance Review", "Lines of Code", "Burnout"). "type": Metric, Framework, Practice, Concept, or Wellbeing. "stance": the author's attitude toward it here (positive, negative, neutral, mixed).
- Return ONLY the JSON object.

Example post: "My manager only cares about story points. Our velocity dropped and now we're being micromanaged. It's burning me out."
{"items":[{"surface":"story points","canonical":"Story Points","type":"Metric","stance":"negative"},{"surface":"velocity","canonical":"Velocity","type":"Metric","stance":"negative"},{"surface":"micromanaged","canonical":"Micromanagement","type":"Practice","stance":"negative"},{"surface":"burning me out","canonical":"Burnout","type":"Wellbeing","stance":"negative"}]}

Example post: "Grinding leetcode for FAANG interviews while job hunting. My side projects help but my career feels stuck and I'm anxious about it."
{"items":[{"surface":"anxious","canonical":"Anxiety","type":"Wellbeing","stance":"negative"}]}

Example post: "What's the best mechanical keyboard for long coding sessions?"
{"items":[]}"""

# Expected output shape (documented; NOT enforced as a grammar — schema-constrained
# sampling was ~6x slower and the model emits valid JSON on its own at temperature 0):
#   {"items":[{"surface","canonical","type","stance"}, ...]}
import re

_TYPES = {"metric", "framework", "practice", "concept", "wellbeing"}
_STANCES = {"positive", "negative", "neutral", "mixed"}


def parse_items(content):
    """Best-effort parse of the model's JSON output into a clean item list."""
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return []
    obj = json.loads(m.group(0))  # raises on malformed -> handled by caller
    out = []
    for it in obj.get("items", []):
        if not isinstance(it, dict):
            continue
        canon = (it.get("canonical") or it.get("surface") or "").strip()
        if not canon:
            continue
        typ = str(it.get("type", "Concept")).strip().capitalize()
        sta = str(it.get("stance", "neutral")).strip().lower()
        out.append({
            "surface": str(it.get("surface", "")).strip(),
            "canonical": canon,
            "type": typ if typ.lower() in _TYPES else "Concept",
            "stance": sta if sta in _STANCES else "neutral",
        })
    return out


def load_done_ids(path):
    """Read existing JSONL, return set of done ids, and rewrite the file keeping only
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
            if "id" in obj:
                done.add(obj["id"])
                valid.append(line)
    # compact rewrite (atomic) so appends continue cleanly
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
    ap.add_argument("--model", required=True, help="path to Gemma 3 GGUF")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--corpus", default=None, help="corpus CSV (default: newest data/*_corpus.csv)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N posts (0 = all)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="only query_group == metrics_productivity")
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all on GPU, 0 = CPU")
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    corpus = args.corpus or max(glob.glob("data/*_corpus.csv"), key=os.path.getctime)
    df = pd.read_csv(corpus, dtype={"title": str, "selftext": str}, low_memory=False)
    df = df[df["type"] == "post"].copy()
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    if args.metrics_only:
        df = df[df["query_group"] == "metrics_productivity"]
    print(f"corpus: {corpus}  | candidate posts: {len(df):,}")

    os.makedirs("data", exist_ok=True)
    done = load_done_ids(args.out)
    todo = df[~df["id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"already done: {len(done):,}  | to process now: {len(todo):,}")
    if todo.empty:
        print("nothing to do.")
        return

    # meta sidecar (written once)
    meta_path = args.out.replace(".jsonl", "_meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"model": os.path.basename(args.model), "corpus": corpus,
                       "prompt_version": 1, "temperature": 0,
                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)

    from llama_cpp import Llama  # imported here so --help works without the dep
    print("loading model...", flush=True)
    llm = Llama(model_path=args.model, n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx, n_threads=args.n_threads, verbose=False)

    t0 = time.time()
    n = n_items = n_err = 0
    total = len(todo)
    bar = tqdm(total=total, unit="post", smoothing=0.05) if tqdm else None
    with open(args.out, "a", encoding="utf-8") as out:
        for row in todo.itertuples(index=False):
            text = (row.title + "\n" + row.selftext).strip()[:2000]
            try:
                resp = llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": "Post:\n" + text}],
                    temperature=0.0, max_tokens=384,
                )
                content = resp["choices"][0]["message"]["content"]
                items = parse_items(content)
                rec = {"id": row.id, "subreddit": row.subreddit,
                       "query_group": row.query_group, "items": items}
                n_items += len(items)
            except Exception as e:  # never crash the batch; mark and continue
                rec = {"id": row.id, "subreddit": row.subreddit,
                       "query_group": row.query_group, "items": [], "error": str(e)[:200]}
                n_err += 1

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())  # durable before next post
            n += 1
            if bar:
                bar.update(1)
                bar.set_postfix(items=n_items, err=n_err,
                                done=f"{len(done)+n:,}/{len(done)+total:,}")
            elif n % 50 == 0 or n == total:  # fallback when tqdm not installed
                rate = n / (time.time() - t0)
                eta_h = (total - n) / rate / 3600 if rate else 0
                print(f"  {n:,}/{total:,}  {rate:.2f} post/s  items={n_items}  "
                      f"err={n_err}  ETA {eta_h:.1f}h", flush=True)
    if bar:
        bar.close()
    print(f"done. processed {n:,} posts ({n_items} items, {n_err} errors) in "
          f"{(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
