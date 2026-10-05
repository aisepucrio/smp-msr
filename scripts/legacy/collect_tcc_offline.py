import argparse
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import zstandard as zstd

SUBREDDITS = [
    "cscareerquestions",
    "ExperiencedDevs",
    "womenintech",
    "softwareengineering",
    "devops",
]

# o dataset usa casing diferente pra alguns nomes de arquivo
FILE_NAME = {
    "softwareengineering": "SoftwareEngineering",
}

QUERIES = {
    "mental_health": (
        "burnout OR exhausted OR overwhelmed OR anxiety OR stress "
        "OR fear OR \"mental health\" OR depression OR therapy "
        "OR medication OR overtime OR quitting OR micromanagement "
        "OR \"imposter syndrome\" OR \"work-life\" OR \"burned out\" "
        "OR toxic OR hostile OR RTO OR \"return to office\""
    ),
    # rede ampla — catalogar métricas específicas é trabalho do analyze_metrics.py
    "metrics_productivity": (
        "metric OR metrics OR measure OR measurement OR OKR OR KPI OR "
        "DORA OR velocity OR \"story points\" OR \"code coverage\" OR "
        "\"lines of code\" OR \"cycle time\" OR throughput OR MTTR OR "
        "SLA OR SLO OR SLI OR burndown OR kanban OR scrum OR "
        "\"tech debt\" OR \"technical debt\" OR \"lead time\" OR "
        "\"deployment frequency\" OR \"change failure\" OR \"pull request\" OR "
        "\"performance review\" OR productivity OR JIRA OR sprint OR "
        "\"four key\" OR \"flow metric\" OR benchmark OR tracking"
    ),
}

DEFAULT_LIMIT = 10
DEFAULT_COMMENTS = 3
DEFAULT_YEAR_START = 2021
DEFAULT_YEAR_END = 2024


def parse_query(query_str):
    terms = []
    for m in re.finditer(r'"([^"]+)"|(\S+)', query_str):
        term = (m.group(1) or m.group(2)).lower()
        if term != "or":
            terms.append(term)
    return terms


def matches(post, terms):
    text = (post.get("title", "") + " " + post.get("selftext", "")).lower()
    return any(t in text for t in terms)


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


def zst_path(data_dir, subreddit, kind):
    name = FILE_NAME.get(subreddit.lower(), subreddit)
    return os.path.join(data_dir, "reddit", "subreddits24", f"{name}_{kind}.zst")


def ts(utc):
    return datetime.fromtimestamp(int(utc), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def collect_posts(data_dir, subreddit, query_name, terms, year_start, year_end, limit):
    path = zst_path(data_dir, subreddit, "submissions")
    if not os.path.exists(path):
        print(f"  [aviso] não encontrado: {path}")
        return []

    rows = []
    for post in stream_zst(path):
        created = int(post.get("created_utc", 0))
        year = datetime.fromtimestamp(created, tz=timezone.utc).year
        if not (year_start <= year <= year_end):
            continue

        selftext = (post.get("selftext", "") or "").strip()
        if selftext in {"[deleted]", "[removed]", ""} and post.get("score", 0) <= 0:
            continue

        if not matches(post, terms):
            continue

        rows.append({
            "id": post.get("id", ""),
            "subreddit": subreddit,
            "query_group": query_name,
            "title": post.get("title", ""),
            "author": post.get("author", "[deleted]"),
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio"),
            "num_comments": post.get("num_comments", 0),
            "created_utc": ts(created),
            "selftext": selftext[:500],
            "permalink": f"https://reddit.com{post.get('permalink', '')}",
            "flair": post.get("link_flair_text") or "",
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def collect_comments(data_dir, subreddit, post_ids, limit_per_post):
    path = zst_path(data_dir, subreddit, "comments")
    if not os.path.exists(path):
        print(f"  [aviso] não encontrado: {path}")
        return []

    counts = defaultdict(int)
    rows = []
    for comment in stream_zst(path):
        raw = comment.get("link_id", "")
        post_id = raw[3:] if raw.startswith("t3_") else raw
        if post_id not in post_ids or counts[post_id] >= limit_per_post:
            continue
        rows.append({
            "post_id": post_id,
            "comment_id": comment.get("id", ""),
            "author": comment.get("author", "[deleted]"),
            "score": comment.get("score", 0),
            "body": (comment.get("body", "") or "")[:500],
            "created_utc": ts(comment.get("created_utc", 0)),
        })
        counts[post_id] += 1
    return rows


def save_csv(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {len(data)} linhas em {path}")


def main():
    parser = argparse.ArgumentParser(description="TCC Reddit Offline Collector")
    parser.add_argument("--data-dir", required=True, help="Diretório com os .zst do Academic Torrents")
    parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
    parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Posts por subreddit por grupo")
    parser.add_argument("--comments", action="store_true", help=f"Top {DEFAULT_COMMENTS} comentários por post")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== TCC Reddit Offline Collector ===")
    print(f"Data dir    : {args.data_dir}")
    print(f"Subreddits  : {', '.join(SUBREDDITS)}")
    print(f"Grupos      : {', '.join(QUERIES.keys())}")
    print(f"Período     : {args.year_start}–{args.year_end}")
    print(f"Limite      : {args.limit} posts por subreddit por grupo")
    print(f"Comentários : {'sim' if args.comments else 'não'}")
    print()

    if args.dry_run:
        print("dry run — nenhum arquivo lido.")
        for name, q in QUERIES.items():
            terms = parse_query(q)
            print(f"\n[{name}] {len(terms)} termos: {terms[:8]}{'...' if len(terms) > 8 else ''}")
        return

    queries_parsed = {name: parse_query(q) for name, q in QUERIES.items()}
    all_posts = []

    for sub in SUBREDDITS:
        for query_name, terms in queries_parsed.items():
            print(f"Lendo r/{sub} [{query_name}] ...", end=" ", flush=True)
            posts = collect_posts(args.data_dir, sub, query_name, terms,
                                  args.year_start, args.year_end, args.limit)
            print(f"{len(posts)} posts")
            all_posts.extend(posts)

    seen = set()
    unique = [p for p in all_posts if not (seen.add(p["id"]) or p["id"] in seen - {p["id"]})]
    # deduplica mantendo ordem
    seen = set()
    unique = []
    for p in all_posts:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    print(f"\nTotal único: {len(unique)} posts ({len(all_posts)} brutos)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(all_posts, f"data/{stamp}_posts_raw.csv")
    save_csv(unique, f"data/{stamp}_posts_unique.csv")

    if args.comments:
        print("\nbuscando comentários...")
        by_sub = defaultdict(set)
        for p in unique:
            by_sub[p["subreddit"]].add(p["id"])

        all_comments = []
        for sub, post_ids in by_sub.items():
            comments = collect_comments(args.data_dir, sub, post_ids, DEFAULT_COMMENTS)
            print(f"  r/{sub}: {len(comments)} comentários")
            all_comments.extend(comments)

        print(f"total: {len(all_comments)} comentários")
        save_csv(all_comments, f"data/{stamp}_comments.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
