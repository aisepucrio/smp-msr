"""
Synthetic corpus generator — SMOKE TESTS ONLY.

Emits a CSV with exactly the schema produced by scripts/build_corpus.py, so the
analysis pipeline can be exercised end-to-end without the ~TB Pushshift dumps:

    id, type, subreddit, query_group, title, selftext, author_hash, score,
    num_comments, created_utc, link_id

    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --posts 800 --comments-per-post 3 --seed 7
    python scripts/generate_mock_data.py --roberta          # also emit a fake labels file

Output goes to data/mock/ by default, which analyze_corpus.py never globs
(it reads data/*_corpus.csv), so the synthetic corpus can never be mistaken for
the real one and real figures/CSVs are never overwritten.

IMPORTANT: the framing/negativity signal below is SYNTHETIC and has no statistical
meaning. It exists only so that every branch of analyze_corpus.py runs with
non-empty groups (all three framing categories, both message types, multiple
threads, all five subreddits, all four years). Never report a number derived
from this file.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timedelta, timezone

SUBREDDITS = ["cscareerquestions", "ExperiencedDevs", "womenintech",
              "softwareengineering", "devops"]

# Mirrors the three framing categories of analyze_corpus.py RX, so all three
# populate. p_neg is tuned only to make the smoke-test groups distinguishable.
FRAMINGS = {
    "dora": {
        "p_neg": 0.18,
        "terms": ["DORA metrics", "deployment frequency", "change failure rate",
                  "mean time to recovery", "the four key metrics",
                  "lead time for changes"],
    },
    "individual": {
        "p_neg": 0.40,
        "terms": ["PIP", "performance review", "story points", "velocity",
                  "OKRs", "KPIs", "lines of code", "commit counts"],
    },
    "scrum": {
        "p_neg": 0.38,
        "terms": ["scrum", "agile", "sprint ceremonies", "the agile process"],
    },
}

NEGATIVE = [
    "this is absolutely brutal and draining every single day",
    "awful, stressful and completely unfair to the people doing the work",
    "a toxic mess that punishes everyone who has to live with it",
    "depressing, exhausting and a joke — burnout is at an all time high",
    "hideous practice, we hate it and our anxiety is through the roof",
]
POSITIVE = [
    "this works really well and is genuinely useful for the whole team",
    "a great, healthy approach that respects everyone involved",
    "excellent, it improved delivery and morale at the same time",
    "a solid win — we are happy and the team feels trusted",
]
NEUTRAL = [
    "we adopted {term} last quarter and wrote it down in the handbook",
    "curious how other teams handle {term} in practice",
    "here is a write up of our experience with {term} over two years",
    "short question about {term} and how it is normally reported",
]
MENTAL_HEALTH_CLAUSE = [
    " Addendum: I am completely burnt out and my manager does not care.",
    " Also, the stress at work has been overwhelming since the layoffs.",
    " Honestly this is why my mental health is in terrible shape.",
    " Context: two colleagues quit from exhaustion last month.",
]


def build_corpus(posts_per_group, comments_per_post, seed, out_path, start=2021, end=2024):
    rng = random.Random(seed)
    authors = [f"user_{i:04d}" for i in range(400)]

    # stable id counter: posts get ids like real Reddit base36, comments hang off them
    def new_id(n):
        return f"t3_{n:07d}"

    window_days = (datetime(end, 12, 31) - datetime(start, 1, 1)).days
    rows = []
    post_ids = []
    n = 0

    def emit(rng, kind, body, subreddit, group, link_id=""):
        nonlocal n
        n += 1
        author = rng.choice(authors)
        when = datetime(start, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=rng.randint(0, window_days), seconds=rng.randint(0, 86399))
        title, selftext = (body, "") if kind == "post" else ("", body)
        return {
            "id": new_id(n),
            "type": kind,
            "subreddit": subreddit,
            "query_group": group,
            "title": title,
            "selftext": selftext,
            "author_hash": hashlib.sha256(author.encode()).hexdigest()[:16],
            "score": rng.randint(0, 900),
            "num_comments": rng.randint(0, 60) if kind == "post" else "",
            "created_utc": when.strftime("%Y-%m-%d %H:%M"),
            "link_id": link_id,
        }

    def sentence(rng):
        framing = rng.choice(list(FRAMINGS))
        term = rng.choice(FRAMINGS[framing]["terms"])
        u = rng.random()
        if u < FRAMINGS[framing]["p_neg"]:
            body = rng.choice(NEGATIVE)
        elif u < FRAMINGS[framing]["p_neg"] + 0.32:
            body = rng.choice(POSITIVE)
        else:
            body = rng.choice(NEUTRAL).format(term=term)
        if rng.random() < 0.45:
            body += rng.choice(MENTAL_HEALTH_CLAUSE)
        # group label mirrors build_corpus.classify(): MH terms take precedence
        group = "mental_health" if rng.random() < 0.55 else "metrics_productivity"
        return framing, body, group

    # --- posts ---
    for _ in range(posts_per_group * len(FRAMINGS)):
        sub = rng.choice(SUBREDDITS)
        _framing, body, group = sentence(rng)
        row = emit(rng, "post", body, sub, group)
        rows.append(row)
        post_ids.append(row["id"])

    # --- comments, nested under real posts so thread clustering has something to cluster ---
    for link in post_ids:
        for _ in range(rng.randint(0, comments_per_post)):
            sub = rng.choice(SUBREDDITS)
            _framing, body, group = sentence(rng)
            rows.append(emit(rng, "comment", body, sub, group, link_id=link))

    cols = ["id", "type", "subreddit", "query_group", "title", "selftext",
            "author_hash", "score", "num_comments", "created_utc", "link_id"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def build_roberta(rows, out_path, seed):
    """Fake transformer labels, purely to exercise analyze_corpus.py section 1c.

    Deterministic pseudo-labels: it just needs to disagree with VADER sometimes
    so the kappa/agreement branch produces non-degenerate numbers.
    """
    rng = random.Random(seed + 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            label = rng.choices(["negative", "neutral", "positive"], weights=[0.35, 0.35, 0.30])[0]
            fh.write(json.dumps({"key": f"{r['type']}:{r['id']}", "label": label}) + "\n")


# --- survey leg -------------------------------------------------------------
# Positional schema expected by scripts/analyze_survey.py (33 columns):
#   2 experience | 3 work_model | 4 team_size | 5 company_size | 6 business
#   7 good_metrics | 8 metrics_used | 9 ideal_metric
#   10 mh_tracking | 11 mh_practices | 12 mh_signals | 13 relatedness
#   15 consent | 16-23 productivity Likert | 24-30 MH-practice Likert | 32 role
PROD_OPTS = [
    ("Story points / Velocity (Scrum)", 0.42, 3.8),
    ("Deployment frequency (quantas vezes por período é feito deploy)", 0.00, None),
    ("Change failure rate / MTTR (DORA metrics)", 0.04, 4.0),
    ("Throughput (número de tarefas/PRs entregues por período)", 0.29, 3.9),
    ("Code coverage / qualidade de código (ex: SonarQube)", 0.17, 4.0),
    ("Performance review / OKRs individuais", 0.29, 4.4),
    ("Linhas de código / número de commits", 0.08, 2.5),
    ("Nenhuma métrica formal", 0.33, 2.5),
    ("Outro", 0.00, None),
]
MH_OPTS = [
    ("Pulse surveys / pesquisas de clima (ex: Office Vibe, Culture Amp)", 0.25, 2.7),
    ("eNPS (Employee Net Promoter Score)", 0.00, None),
    ("1:1s regulares focados em bem-estar", 0.21, 4.4),
    ("Programa de assistência ao empregado (EAP / psicólogo)", 0.04, 4.0),
    ("Benefício de terapia / saúde mental", 0.29, 4.3),
    ('"No meeting days" / proteção de tempo de foco', 0.04, 4.0),
    ("Nenhuma prática formal", 0.46, 2.5),
    ("Outro", 0.04, None),
]
ROLES = ["Desenvolvedor(a) back-end", "Desenvolvedor(a) front-end",
         "Desenvolvedor(a) full-stack", "Engenheiro(a) de dados/IA",
         "Analista de negócios", "Tech Lead", "Gerente de engenharia"]


def build_survey(n, out_path, seed):
    rng = random.Random(seed + 2)
    cols = [f"col{i}" for i in range(33)]
    rows = []
    for _ in range(n):
        row = {f"col{i}": "" for i in range(33)}
        row["col2"] = rng.choice(["0-2 anos", "2-5 anos", "5-10 anos", "10+ anos"])
        row["col3"] = rng.choice(["100% presencial", "Híbrido", "100% remoto"])
        row["col4"] = rng.choice(["1-5 pessoas", "6-10 pessoas", "10-20 pessoas", "20+ pessoas"])
        row["col5"] = rng.choice(["Menos de 20 pessoas", "20-100 pessoas", "100-500 pessoas",
                                  "500-2.000 pessoas", "Mais de 2.000 pessoas"])
        row["col6"] = rng.choice(["Produto", "Serviços", "Consultoria"])
        row["col7"] = rng.choice(["qualidade do código", "colaboração", "proatividade"])
        row["col9"] = rng.choice(["tempo até o valor", "retrabalho", "complexidade cognitiva"])
        row["col12"] = rng.choice(["queda de qualidade", "exaustão", "isolamento"])
        row["col10"] = rng.choice(["Sim, formalmente", "Informalmente", "Não"])
        row["col13"] = rng.choices([4, 5], weights=[0.29, 0.71])[0]
        row["col15"] = "Sim"
        row["col32"] = rng.choice(ROLES)
        # multi-select cells: join the selected options with ", "
        for base, opts, first in (("8", PROD_OPTS, 16), ("11", MH_OPTS, 24)):
            picked = [o for o, p, _ in opts if rng.random() < p]
            row[f"col{base}"] = ", ".join(picked)
            for k, (o, _p, mean) in enumerate(opts):
                if mean is None or o not in picked:
                    continue  # no follow-up slider, or the respondent did not select it
                row[f"col{first + k}"] = max(1, min(5, round(rng.gauss(mean, 0.7))))
        rows.append(row)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posts", type=int, default=120,
                    help="posts per framing category (default: 120)")
    ap.add_argument("--comments-per-post", type=int, default=4,
                    help="max comments per post (default: 4)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: data/mock/<stamp>_corpus.csv)")
    ap.add_argument("--roberta", action="store_true",
                    help="also write a synthetic data/mock/mock_roberta_sentiment.jsonl")
    ap.add_argument("--survey", action="store_true",
                    help="also write a synthetic data/mock/mock_survey_results.csv")
    ap.add_argument("--survey-n", type=int, default=24)
    args = ap.parse_args()

    out = args.out or os.path.join(
        "data", "mock", datetime.now().strftime("%Y%m%d_%H%M%S") + "_corpus.csv")
    rows = build_corpus(args.posts, args.comments_per_post, args.seed, out)
    n_post = sum(1 for r in rows if r["type"] == "post")
    print(f"synthetic corpus: {len(rows):,} rows ({n_post:,} posts, "
          f"{len(rows)-n_post:,} comments) -> {out}")

    if args.roberta:
        rb = os.path.join("data", "mock", "mock_roberta_sentiment.jsonl")
        build_roberta(rows, rb, args.seed)
        print(f"synthetic roberta labels: {len(rows):,} -> {rb}")

    if args.survey:
        sv = os.path.join("data", "mock", "mock_survey_results.csv")
        build_survey(args.survey_n, sv, args.seed)
        print(f"synthetic survey: {args.survey_n} rows -> {sv}")


if __name__ == "__main__":
    main()
