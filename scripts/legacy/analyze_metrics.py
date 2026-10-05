"""
Lê um CSV de posts coletados e cataloga quais métricas de produtividade aparecem.

    python analyze_metrics.py data/20260413_134353_posts_unique.csv
    python analyze_metrics.py data/20260413_134353_posts_unique.csv --csv
"""

import argparse
import re
import sys
from collections import defaultdict

import pandas as pd

# Termos curtos que precisam de word boundary pra não dar falso positivo
BOUNDARY_TERMS = {"sla", "slo", "sli", "kpi", "okr", "pip", "wip", "dora", "mttr", "mttd"}

CATALOG = {
    "DORA (framework)":               ["dora metrics", "dora"],
    "DORA — Deployment Frequency":    ["deployment frequency", "deploys per day"],
    "DORA — Lead Time for Changes":   ["lead time for changes"],
    "DORA — Change Failure Rate":     ["change failure rate", "change failure"],
    "DORA — MTTR":                    ["mttr", "mean time to restore", "mean time to repair"],
    "SPACE Framework":                ["space framework", "space metrics"],
    "Four Key Metrics":               ["four key metrics", "four keys"],
    "OKR":                            ["okr", "objectives and key results"],
    "KPI":                            ["kpi", "key performance indicator"],
    "Velocity":                       ["velocity", "sprint velocity"],
    "Story Points":                   ["story points", "story point"],
    "Cycle Time":                     ["cycle time"],
    "Throughput":                     ["throughput"],
    "Burndown / Burnup":              ["burndown", "burnup"],
    "WIP Limits":                     ["wip limit", "work in progress limit"],
    "Cumulative Flow":                ["cumulative flow"],
    "Code Coverage":                  ["code coverage", "test coverage"],
    "Lines of Code (LOC)":            ["lines of code"],
    "Code Churn":                     ["code churn"],
    "PR / Review Metrics":            ["pr metrics", "pull request metrics", "time to merge",
                                       "review time", "prs per day", "pull requests per day"],
    "Commit Frequency":               ["commits per day", "commit frequency"],
    "SLA / SLO / SLI":               ["service level agreement", "service level objective",
                                       "service level indicator"],
    "Error Budget":                   ["error budget"],
    "Uptime / Availability":          ["uptime", "availability metric", "99.9%", "99.99%"],
    "MTTD":                           ["mttd", "mean time to detect"],
    "Incident Rate":                  ["incident rate", "incident count", "incidents per"],
    "Performance Review":             ["performance review", "perf review", "annual review"],
    "Stack Ranking":                  ["stack ranking", "stack rank", "forced ranking"],
    "PIP":                            ["pip", "performance improvement plan"],
    "360 Feedback":                   ["360 feedback", "360 review", "360-degree"],
    "Technical Debt":                 ["technical debt", "tech debt"],
    "Defect / Bug Rate":              ["defect rate", "bug rate", "escaped defects"],
    "Developer Productivity":         ["developer productivity", "engineering productivity"],
    "Flow Metrics":                   ["flow efficiency", "flow velocity", "flow time", "flow metrics"],
    "Accelerate":                     ["accelerate book", "accelerate metrics"],
    "Alert Fatigue / On-call":        ["alert fatigue", "on-call burden"],
    "eNPS / Satisfaction":            ["enps", "employee nps", "satisfaction score"],
    "Release Frequency":              ["release frequency", "release cadence"],
    "Test Automation %":              ["test automation rate", "automated test coverage"],
    "Sprint Goal":                    ["sprint goal", "sprint review"],
    "1:1 Cadence":                    ["1:1", "one-on-one"],
}


def hit(text, term):
    if term in BOUNDARY_TERMS:
        return bool(re.search(r"\b" + re.escape(term) + r"\b", text))
    return term in text


def find_metrics(text):
    found = []
    for metric, terms in CATALOG.items():
        if any(hit(text, t) for t in terms):
            found.append(metric)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV de posts (ex: data/20260413_posts_unique.csv)")
    parser.add_argument("--csv", dest="save_csv", action="store_true",
                        help="Salva o catálogo como CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "query_group" in df.columns:
        df = df[df["query_group"] == "metrics_productivity"]

    print(f"Analisando {len(df)} posts...\n")

    counts = defaultdict(list)
    for _, row in df.iterrows():
        text = (str(row.get("title", "")) + " " + str(row.get("selftext", ""))).lower()
        for metric in find_metrics(text):
            counts[metric].append((row.get("score", 0), row.get("subreddit", ""), row.get("title", "")))

    if not counts:
        print("Nenhuma métrica encontrada.")
        sys.exit(0)

    print(f"{'Métrica':<40} {'Posts':>5}  Melhor exemplo")
    print("-" * 100)
    for metric, posts in sorted(counts.items(), key=lambda x: -len(x[1])):
        top = max(posts, key=lambda x: x[0])
        print(f"{metric:<40} {len(posts):>5}  [{top[0]:>4}] r/{top[1]}: {top[2][:55]}")

    not_found = [m for m in CATALOG if m not in counts]
    print(f"\n{len(counts)} métricas encontradas, {len(not_found)} não encontradas")
    if not_found:
        print(f"Ausentes: {', '.join(not_found)}")

    if args.save_csv:
        rows = []
        for metric, posts in counts.items():
            for score, sub, title in posts:
                rows.append({"metric": metric, "score": score, "subreddit": sub, "title": title})
        out = args.csv.replace("_unique.csv", "_metrics_catalog.csv").replace(".csv", "_metrics_catalog.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSalvo em {out}")


if __name__ == "__main__":
    main()
