"""
Build the full offline corpus (posts + comments) from the Academic Torrents Reddit
dumps. Unlike the old top-N collector, this keeps EVERY matching document in the
2021-2024 window, tags it post/comment, and streams straight to CSV (memory-safe).

    python scripts/download_offline.py            # fetch the .zst dumps first
    python scripts/build_corpus.py                # -> data/<timestamp>_corpus.csv
    # add --no-comments to build posts only

Output: data/<timestamp>_corpus.csv with columns
    id, type, subreddit, query_group, title, selftext, author_hash,
    score, num_comments, created_utc, link_id
Authors are SHA-256 hashed and permalinks dropped (publishable).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone

import zstandard as zstd

SUBREDDITS = ["cscareerquestions", "ExperiencedDevs", "womenintech",
              "softwareengineering", "devops"]
# The dump spells two of these in mixed case; the mirror/torrent sources vary, so we
# resolve the real filename case-insensitively instead of hardcoding one spelling.
# (Hardcoding one spelling silently dropped r/softwareengineering and r/ExperiencedDevs
# on a case-sensitive filesystem.)
FILE_ALIASES = {"softwareengineering": {"softwareengineering"},
                "ExperiencedDevs": {"experienceddevs"}}

QUERIES = {
    "mental_health": [
        "burnout", "exhausted", "overwhelmed", "anxiety", "stress", "fear",
        "mental health", "depression", "therapy", "medication", "overtime",
        "quitting", "micromanagement", "imposter syndrome", "work-life",
        "burned out", "toxic", "hostile", "rto", "return to office",
    ],
    "metrics_productivity": [
        "metric", "metrics", "measure", "measurement", "okr", "kpi", "dora",
        "velocity", "story points", "code coverage", "lines of code",
        "cycle time", "throughput", "mttr", "sla", "slo", "sli", "burndown",
        "kanban", "scrum", "tech debt", "technical debt", "lead time",
        "deployment frequency", "change failure", "pull request",
        "performance review", "productivity", "jira", "sprint", "four key",
        "flow metric", "benchmark", "tracking",
    ],
}
# whole-word match for short/ambiguous terms to avoid substring false positives
BOUNDARY = {"rto", "okr", "kpi", "dora", "mttr", "sla", "slo", "sli", "jira"}


def build_regex(terms):
    parts = []
    for t in terms:
        t = re.escape(t)
        parts.append(rf"\b{t}\b" if t in BOUNDARY else t)
    return re.compile("|".join(parts), re.IGNORECASE)


REGEX = {g: build_regex(t) for g, t in QUERIES.items()}


def classify(text):
    """Return query_group (mental_health takes precedence) or None."""
    if REGEX["mental_health"].search(text):
        return "mental_health"
    if REGEX["metrics_productivity"].search(text):
        return "metrics_productivity"
    return None


def stream_zst(path):
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def hash_author(a):
    a = a or "[deleted]"
    return hashlib.sha256(a.encode("utf-8")).hexdigest()[:16]


def ts(utc):
    return datetime.fromtimestamp(int(utc), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def in_window(utc, y0, y1):
    return y0 <= datetime.fromtimestamp(int(utc), tz=timezone.utc).year <= y1


def resolve_path(archive_dir, sub, kind):
    """Find <sub>_<kind>.zst in archive_dir, case-insensitively.

    Returns (found_path_or_None, human_readable_list_of_paths_tried). The caller
    decides whether a miss is fatal, so a missing dump is a clear error instead of
    a silent subreddit-sized hole in the corpus.
    """
    names = FILE_ALIASES.get(sub, {sub.lower()})
    tried = [os.path.join(archive_dir, f"{c}_{kind}.zst") for c in sorted(names)]
    for p in tried:
        if os.path.exists(p):
            return p, tried
    return None, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default="data/reddit/subreddits24",
                    help="dir holding <subreddit>_{submissions,comments}.zst "
                         "(default: %(default)s — what download_offline.py writes)")
    ap.add_argument("--year-start", type=int, default=2021)
    ap.add_argument("--year-end", type=int, default=2024)
    ap.add_argument("--no-comments", action="store_true")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"data/{stamp}_corpus.csv"
    cols = ["id", "type", "subreddit", "query_group", "title", "selftext",
            "author_hash", "score", "num_comments", "created_utc", "link_id"]

    counts = {"post": 0, "comment": 0}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()

        for sub in SUBREDDITS:
            # --- posts (submissions) ---
            spath, tried = resolve_path(args.archive_dir, sub, "submissions")
            if spath is None:
                raise SystemExit(
                    f"r/{sub}: submissions dump not found. Tried: "
                    f"{', '.join(tried)}\nDownload it first (see README) or pass "
                    f"--archive-dir.")
            n0 = counts["post"]
            for p in stream_zst(spath):
                utc = p.get("created_utc", 0)
                if not in_window(utc, args.year_start, args.year_end):
                    continue
                self = (p.get("selftext", "") or "").strip()
                if self in {"[deleted]", "[removed]", ""} and p.get("score", 0) <= 0:
                    continue
                g = classify(p.get("title", "") + " " + self)
                if not g:
                    continue
                w.writerow({
                    "id": p.get("id", ""), "type": "post", "subreddit": sub,
                    "query_group": g, "title": p.get("title", ""),
                    "selftext": self[:500], "author_hash": hash_author(p.get("author")),
                    "score": p.get("score", 0), "num_comments": p.get("num_comments", 0),
                    "created_utc": ts(utc), "link_id": "",
                })
                counts["post"] += 1
            print(f"r/{sub}: {counts['post'] - n0} posts")

            # --- comments ---
            if args.no_comments:
                continue
            cpath, tried = resolve_path(args.archive_dir, sub, "comments")
            if cpath is None:
                print(f"  [skip r/{sub} comments] not found; tried: "
                      f"{', '.join(tried)}")
                continue
            n0 = counts["comment"]
            for c in stream_zst(cpath):
                utc = c.get("created_utc", 0)
                if not in_window(utc, args.year_start, args.year_end):
                    continue
                body = (c.get("body", "") or "").strip()
                if body in {"[deleted]", "[removed]", ""}:
                    continue
                g = classify(body)
                if not g:
                    continue
                w.writerow({
                    "id": c.get("id", ""), "type": "comment", "subreddit": sub,
                    "query_group": g, "title": "", "selftext": body[:500],
                    "author_hash": hash_author(c.get("author")),
                    "score": c.get("score", 0), "num_comments": "",
                    "created_utc": ts(utc),
                    "link_id": (c.get("link_id", "") or "").replace("t3_", ""),
                })
                counts["comment"] += 1
            print(f"r/{sub}: {counts['comment'] - n0} comments")

    print(f"\nTotal: {counts['post']} posts, {counts['comment']} comments")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
