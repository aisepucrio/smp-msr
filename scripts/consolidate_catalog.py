"""
Consolidate the open-extraction LLM labels (data/llm_labels.jsonl) into a clean metric
catalog. Canonicalizes the many surface/canonical forms the model emits, merges synonyms
against a seed taxonomy (+ fuzzy matching for the long tail), counts distinct posts per
item, picks a majority type, and flags items NOT in the seed taxonomy as `emerging`
(the newly discovered metrics).

    python scripts/consolidate_catalog.py
    python scripts/consolidate_catalog.py --min-count 3

Output: data/metricas_catalogadas.csv with columns rank,nome,tipo,posts,emerging
(same schema as the regex catalog + an `emerging` column), consumed by analyze_corpus.py.
"""
import argparse
import collections
import json
import re

import pandas as pd
from rapidfuzz import fuzz

# Seed taxonomy: canonical display -> known synonyms (lowercase). Mirrors the regex
# CATALOG in analyze_corpus.py so "known" metrics keep stable names and only genuinely
# new items are flagged emerging.
SEED = {
    "Velocity": ["velocity", "sprint velocity"],
    "Story Points": ["story points", "story point", "sprint points", "story pointing"],
    "DORA": ["dora", "dora metrics", "four key metrics", "four keys"],
    "Deployment Frequency": ["deployment frequency", "deploys per day", "deploy frequency"],
    "Lead Time": ["lead time", "lead time for changes"],
    "MTTR": ["mttr", "mean time to recovery", "mean time to restore", "mean time to repair"],
    "Change Failure Rate": ["change failure rate", "change failure"],
    "Lines of Code": ["lines of code", "loc", "line of code"],
    "Code Coverage": ["code coverage", "test coverage"],
    "Cycle Time": ["cycle time"],
    "Throughput": ["throughput"],
    "OKR": ["okr", "okrs", "objectives and key results"],
    "KPI": ["kpi", "kpis", "key performance indicator", "key performance indicators"],
    "Performance Review": ["performance review", "perf review", "annual review",
                           "performance evaluation", "performance reviews"],
    "PIP": ["pip", "performance improvement plan"],
    "Stack Ranking": ["stack ranking", "stack rank", "forced ranking"],
    "Technical Debt": ["technical debt", "tech debt"],
    "Developer Productivity": ["developer productivity", "engineering productivity"],
    "Uptime / Availability": ["uptime", "availability", "availability metric"],
    "SLA / SLO / SLI": ["sla", "slo", "sli", "service level agreement",
                        "service level objective", "service level indicator"],
    "Error Budget": ["error budget"],
    "1:1 Cadence": ["1:1", "1:1s", "one-on-one", "one on one", "1 on 1"],
    "Sprint Goal": ["sprint goal", "sprint review"],
    "Burnout": ["burnout", "burned out", "burning out", "burnt out"],
    "Micromanagement": ["micromanagement", "micromanaged", "micromanaging", "micro-management"],
    "eNPS": ["enps", "employee nps", "employee net promoter score"],
    "Pulse Survey": ["pulse survey", "pulse surveys"],
    "Number of Commits": ["number of commits", "commit count", "commits per day",
                          "commit frequency"],
    "Pull Requests": ["pull request", "pull requests", "pr count", "prs merged",
                      "pr throughput"],
}
SEED_TYPE = {
    "Velocity": "Metric", "Story Points": "Metric", "DORA": "Framework",
    "Deployment Frequency": "Metric", "Lead Time": "Metric", "MTTR": "Metric",
    "Change Failure Rate": "Metric", "Lines of Code": "Metric", "Code Coverage": "Metric",
    "Cycle Time": "Metric", "Throughput": "Metric", "OKR": "Framework", "KPI": "Concept",
    "Performance Review": "Practice", "PIP": "Practice", "Stack Ranking": "Practice",
    "Technical Debt": "Concept", "Developer Productivity": "Concept",
    "Uptime / Availability": "Metric", "SLA / SLO / SLI": "Metric", "Error Budget": "Metric",
    "1:1 Cadence": "Practice", "Sprint Goal": "Practice", "Burnout": "Wellbeing",
    "Micromanagement": "Practice", "eNPS": "Metric", "Pulse Survey": "Practice",
    "Number of Commits": "Metric", "Pull Requests": "Metric",
}


def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[\s\-_/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s).strip()
    return s


# Stoplist: canonical forms the open-extraction model emits that are NOT productivity
# metrics, evaluation practices, agile frameworks, or well-being indicators, and so are
# dropped from the catalog. Three buckets: programming languages, infra/observability
# tooling (the tool itself, not a metric), and interview-prep / generic nouns the
# prompt's exclusion list did not fully suppress. Matched on norm(), so case/punctuation
# /spacing-insensitive. Edit freely — this is the main precision knob for the catalog.
STOP = {norm(s) for s in [
    # programming / query languages
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "go", "golang",
    "rust", "ruby", "php", "kotlin", "swift", "scala", "node", "nodejs", "html", "css",
    "react", "angular", "vue", ".net", "dotnet",
    # infra / cloud / observability / dev tools
    "kubernetes", "docker", "terraform", "prometheus", "grafana", "jenkins", "datadog",
    "aws", "azure", "gcp", "git", "github", "gitlab", "slack", "chatgpt", "copilot",
    "microservices",
    # interview-prep / job-hunt noise the user asked to exclude
    "leetcode", "algorithms", "system design", "interviews", "interview", "faang",
    "gpa", "resume",
    # generic nouns / off-topic that slipped past the prompt exclusions
    "time", "hours", "metrics", "productivity", "performance", "experience",
    "years of experience", "salary", "compensation", "pay", "raises", "bonus", "money",
    "project", "projects", "tasks", "task", "code", "codebase", "bug", "bugs",
    "features", "feature", "backlog", "stories", "story", "requirements", "skills",
    "skill", "team", "teams", "developers", "developer", "junior", "senior",
    "percentage", "management", "expectations", "communication", "feedback",
    "learning", "training", "frontend", "backend", "product owner",
]}


def build_seed_lookup():
    lut = {}
    for canon, syns in SEED.items():
        lut[norm(canon)] = canon
        for s in syns:
            lut[norm(s)] = canon
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/llm_labels.jsonl")
    ap.add_argument("--out", default="data/metricas_catalogadas.csv")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--fuzzy-seed", type=int, default=92, help="match-to-seed threshold")
    ap.add_argument("--fuzzy-emerge", type=int, default=90, help="merge-emerging threshold")
    args = ap.parse_args()

    seed_lut = build_seed_lookup()
    seed_norms = list(seed_lut.keys())

    # canonical_key -> {posts:set, types:Counter, display:str, emerging:bool}
    cat = {}
    emerging_reps = []  # list of (norm_key, canonical_key) for fuzzy clustering

    def resolve(canon_raw):
        n = norm(canon_raw)
        if not n or n in STOP:
            return None, None
        # 1) exact seed
        if n in seed_lut:
            return seed_lut[n], False
        # 2) fuzzy seed
        best, score = None, 0
        for sn in seed_norms:
            r = fuzz.ratio(n, sn)
            if r > score:
                best, score = sn, r
        if score >= args.fuzzy_seed:
            return seed_lut[best], False
        # 3) fuzzy against existing emerging reps
        for rep_norm, rep_key in emerging_reps:
            if fuzz.token_sort_ratio(n, rep_norm) >= args.fuzzy_emerge:
                return rep_key, True
        # 4) brand new emerging
        display = canon_raw.strip().title() if canon_raw.islower() else canon_raw.strip()
        emerging_reps.append((n, display))
        return display, True

    n_recs = n_items = 0
    with open(args.labels, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_recs += 1
            pid = rec.get("id")
            for it in rec.get("items", []):
                canon_raw = it.get("canonical") or it.get("surface")
                key, emerging = resolve(canon_raw)
                if key is None:
                    continue
                n_items += 1
                e = cat.setdefault(key, {"posts": set(), "types": collections.Counter(),
                                         "emerging": emerging})
                e["posts"].add(pid)
                e["types"][it.get("type", "Concept")] += 1

    rows = []
    for key, e in cat.items():
        cnt = len(e["posts"])
        if cnt < args.min_count:
            continue
        tipo = (SEED_TYPE.get(key)
                or (e["types"].most_common(1)[0][0] if e["types"] else "Concept"))
        rows.append({"nome": key, "tipo": tipo, "posts": cnt,
                     "emerging": bool(e["emerging"])})
    out = (pd.DataFrame(rows).sort_values("posts", ascending=False)
           .reset_index(drop=True))
    out.insert(0, "rank", out.index + 1)
    out.to_csv(args.out, index=False)

    print(f"labels read: {n_recs:,} posts, {n_items:,} extracted items")
    print(f"catalog: {len(out)} items ({out['emerging'].sum()} emerging) -> {args.out}\n")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
