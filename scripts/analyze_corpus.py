"""
Single source of truth for the thesis results: reads the offline corpus
(posts + comments), prints every number used in the text/tables, and regenerates
all figures. Run from repo root:

    python scripts/analyze_corpus.py

Reads the most recent data/*_corpus.csv. Figures -> docs/figures/.

Every input and output path is overridable, which is what lets the smoke test
(`make smoke`) run the whole pipeline against a synthetic corpus in data/mock/
without touching the real corpus or the published figures:

    python scripts/analyze_corpus.py --corpus data/mock/mock_corpus.csv \
        --out data/mock/figures --roberta data/mock/mock_roberta_sentiment.jsonl
"""
import argparse
import glob
import itertools
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
import statsmodels.formula.api as smf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ap = argparse.ArgumentParser(description="Analyse the Reddit corpus (thesis results).")
_ap.add_argument("--corpus", default=None,
                 help="corpus CSV (default: newest data/*_corpus.csv)")
_ap.add_argument("--out", default="docs/figures", help="figure output directory")
_ap.add_argument("--roberta", default="data/roberta_sentiment.jsonl",
                 help="second-model labels; skipped if missing")
_ap.add_argument("--curated-catalog", default="data/metricas_catalogadas_filtradas.csv",
                 help="curated LLM catalog (catalog of record for Fig. 2)")
_ap.add_argument("--full-catalog", default="data/metricas_catalogadas.csv",
                 help="un-curated LLM catalog (used only for coverage validation)")
_ap.add_argument("--metrics-csv", default="data/metricas_analise_sentimento.csv",
                 help="per-metric table output")
_args = _ap.parse_args()

OUT = _args.out
os.makedirs(OUT, exist_ok=True)
AN = SentimentIntensityAnalyzer()

PALETTE = {
    "mental_health": "#c0392b", "metrics_productivity": "#2980b9",
    "dora": "#27ae60", "individual": "#e67e22", "scrum": "#8e44ad",
}

# ---------- load ----------
if _args.corpus:
    path = _args.corpus
    if not os.path.exists(path):
        raise SystemExit(f"--corpus not found: {path}")
else:
    csvs = glob.glob("data/*_corpus.csv")
    if not csvs:
        raise SystemExit(
            "No data/*_corpus.csv found. Run scripts/build_corpus.py first "
            "(see README) or pass --corpus explicitly.")
    path = max(csvs, key=os.path.getctime)
print(f"corpus: {path}\n")
df = pd.read_csv(path, dtype={"title": str, "selftext": str, "id": str, "link_id": str},
                 low_memory=False)
df["title"] = df["title"].fillna("")
df["selftext"] = df["selftext"].fillna("")
df["text"] = (df["title"] + " " + df["selftext"]).str.lower()
df["dt"] = pd.to_datetime(df["created_utc"], errors="coerce")
df["year"] = df["dt"].dt.year
df["month"] = df["dt"].dt.to_period("M")
# comments are nested in threads: cluster id = the parent post (link_id has the
# t3_ prefix already stripped by build_corpus.py), posts are their own thread
df["thread_id"] = df["link_id"].where(df["type"] == "comment", df["id"]).fillna(df["id"])

posts = df[df["type"] == "post"].copy()
comments = df[df["type"] == "comment"].copy()


def vader(series):
    return series.apply(lambda x: AN.polarity_scores(x)["compound"])


def stats(frame):
    """mean sentiment, negativity rate, N for a frame (expects 'compound')."""
    n = len(frame)
    if n == 0:
        return 0.0, 0.0, 0
    return frame["compound"].mean(), (frame["compound"] < -0.05).mean(), n


# ---------- 0. overview ----------
print("=" * 60)
print("CORPUS OVERVIEW")
print(f"documents: {len(df):,}  (posts {len(posts):,}, comments {len(comments):,})")
print("posts by group   :", posts["query_group"].value_counts().to_dict())
print("comments by group:", comments["query_group"].value_counts().to_dict())
print("posts by subreddit:", posts["subreddit"].value_counts().to_dict())

# ---------- 1. metric framing masks ----------
RX = {
    "dora": re.compile(r"\bdora\b|deployment frequency|change failure|"
                       r"mean time to recover|four key", re.I),
    "individual": re.compile(r"\bpip\b|performance improvement plan|performance review|"
                             r"\bstory points?\b|\bvelocity\b|\bokrs?\b|\bkpis?\b|"
                             r"lines of code", re.I),
    "scrum": re.compile(r"\bscrum\b|\bagile\b", re.I),
}


def category_frame(rx):
    sub = df[df["text"].str.contains(rx)].copy()
    sub["compound"] = vader(sub["text"])
    return sub


print("\n" + "=" * 60)
print("SENTIMENT BY FRAMING (full corpus: posts + comments)")
print("Table 4.2 columns -> Full: N / Sent / Neg ; Posts: N / Sent / Neg ; Comments: N / Sent / Neg")
framing_rows = []
cat_frames = {}
for cat, rx in RX.items():
    f = category_frame(rx)
    cat_frames[cat] = f
    m, neg, n = stats(f)
    # posts-only and comments-only robustness
    mp, negp, npost = stats(f[f["type"] == "post"])
    mc, negc, ncom = stats(f[f["type"] == "comment"])
    framing_rows.append((cat, n, m, neg, npost, mp, negp, ncom, mc, negc))
    print(f"{cat:11s} | full N={n:6d} sent={m:+.3f} neg={neg*100:4.1f}%"
          f" | posts N={npost:5d} sent={mp:+.3f} neg={negp*100:4.1f}%"
          f" | comments N={ncom:6d} sent={mc:+.3f} neg={negc*100:4.1f}%")

# --- significance test on the framing categories (full corpus) ---
# NB: the three framing masks are NOT mutually exclusive (a doc mentioning both
# "velocity" and "dora" is counted in both), so the chi-square independence assumption
# is approximate; we report the pairwise overlap to show it is small, and also give the
# central pairwise test (DORA vs Individual) on its disjoint part.
print("\n" + "-" * 60)
print("SIGNIFICANCE (framing negativity & sentiment)")
neg_counts = {c: int((cat_frames[c]["compound"] < -0.05).sum()) for c in RX}
pos_counts = {c: len(cat_frames[c]) - neg_counts[c] for c in RX}
cont = np.array([[neg_counts[c], pos_counts[c]] for c in RX])  # 3x2
chi2, p, dof, _ = chi2_contingency(cont)
# Cramer's V as the effect size: with N in the tens of thousands a tiny gap is
# "significant", so we report the magnitude alongside p. V = sqrt(chi2 / (N*(k-1))),
# k=min(rows,cols)=2, so V reduces to sqrt(chi2/N) here.
N3 = int(cont.sum())
cramer_v = (chi2 / (N3 * 1)) ** 0.5
print(f"  chi-square 3x2 (category x negative): chi2={chi2:.1f}, dof={dof}, "
      f"p={p:.2e}, N={N3:,}, Cramer's V={cramer_v:.3f}")
# central comparison: DORA vs Individual (2x2)
cont2 = np.array([[neg_counts['dora'], pos_counts['dora']],
                  [neg_counts['individual'], pos_counts['individual']]])
chi2b, p2, dof2, _ = chi2_contingency(cont2)
N2 = int(cont2.sum())
phi = (chi2b / N2) ** 0.5
print(f"  chi-square 2x2 (DORA vs Individual):  chi2={chi2b:.1f}, dof={dof2}, "
      f"p={p2:.2e}, N={N2:,}, phi={phi:.3f}")
# Kruskal-Wallis on the compound distributions
H, pk = kruskal(*[cat_frames[c]["compound"].values for c in RX])
print(f"  Kruskal-Wallis (compound across 3):   H={H:.1f}, p={pk:.2e}")
# pairwise overlaps (for the threats note)
for a, b in [("dora", "individual"), ("dora", "scrum"), ("individual", "scrum")]:
    ov = len(set(cat_frames[a].index) & set(cat_frames[b].index))
    print(f"  overlap {a}∩{b}: {ov:,} docs "
          f"({ov/len(cat_frames[a])*100:.1f}% of {a})")
# posts-only 2x2: each post is its own thread, so no nesting at all
post_counts = {}
for c in ("dora", "individual"):
    fp = cat_frames[c][cat_frames[c]["type"] == "post"]
    post_counts[c] = (int((fp["compound"] < -0.05).sum()), len(fp))
cont_p = np.array([[neg, tot - neg] for neg, tot in post_counts.values()])
chi2p, pp, dofp, _ = chi2_contingency(cont_p)
Np = int(cont_p.sum())
print(f"  chi-square 2x2 POSTS ONLY (DORA vs Individual): chi2={chi2p:.1f}, dof={dofp}, "
      f"p={pp:.2e}, N={Np:,}, phi={(chi2p / Np) ** 0.5:.3f}")

# ---------- 1b. logistic regression (primary inference) ----------
# The chi-square/KW tests above treat every message as independent, but comments
# are nested in threads, which inflates the effective N. The primary inference is
# therefore a logistic regression of negativity on the framing, controlling for
# subreddit, year, message type and length, with standard errors clustered by
# thread (Cameron & Miller 2015). Restricted to docs matching exactly one framing
# so the framing is a clean categorical.
print("\n" + "-" * 60)
print("REGRESSION  neg ~ framing + subreddit + year + is_comment + log_len")
print("            (SEs clustered by thread; docs matching exactly one framing)")

allf = pd.concat([cat_frames[c].assign(framing=c) for c in RX])
excl = allf[~allf.index.duplicated(keep=False)].copy()
n_union = allf.index.nunique()
print(f"  framing union: {n_union:,} docs | exactly one framing: {len(excl):,} "
      f"(excluded {n_union - len(excl):,} multi-framing docs)")


def prep_logit(frame):
    out = frame.dropna(subset=["year", "thread_id"]).copy()
    out["year"] = out["year"].astype(int)
    out["neg"] = (out["compound"] < -0.05).astype(int)
    out["is_comment"] = (out["type"] == "comment").astype(int)
    out["log_len"] = np.log(out["text"].str.len() + 1)
    return out


def framing_logit(data, outcome):
    """Cluster-robust logit of a binary outcome on framing + controls."""
    m = smf.logit(
        f"{outcome} ~ C(framing, Treatment('individual')) + C(subreddit) + C(year)"
        " + is_comment + log_len", data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["thread_id"]}, disp=0)
    ors, ci = np.exp(m.params), np.exp(m.conf_int())
    for c in RX:
        term = f"C(framing, Treatment('individual'))[T.{c}]"
        if term in m.params.index:
            print(f"    {c:5s} vs individual: OR={ors[term]:.3f} "
                  f"[{ci.loc[term, 0]:.3f}, {ci.loc[term, 1]:.3f}]  p={m.pvalues[term]:.2e}")
    print(f"    N={int(m.nobs):,} docs | {data['thread_id'].nunique():,} threads | "
          f"pseudo-R2={m.prsquared:.3f}")
    return m


excl = prep_logit(excl)
framing_logit(excl, "neg")

# robustness: full union with non-exclusive 0/1 dummies (individual-only baseline)
uni = prep_logit(allf[~allf.index.duplicated(keep="first")])
for c in RX:
    uni[c] = uni.index.isin(cat_frames[c].index).astype(int)
mu = smf.logit("neg ~ dora + scrum + C(subreddit) + C(year) + is_comment + log_len",
               data=uni).fit(cov_type="cluster", cov_kwds={"groups": uni["thread_id"]}, disp=0)
print(f"  robustness (full union, non-exclusive dummies): "
      f"dora OR={np.exp(mu.params['dora']):.3f} p={mu.pvalues['dora']:.2e}")

# ---------- 1c. second sentiment model (VADER vs RoBERTa) ----------
ROBERTA = _args.roberta
print("\n" + "=" * 60)
print("VADER vs RoBERTa AGREEMENT (framing-matched docs)")
if not os.path.exists(ROBERTA):
    print(f"[skip] {ROBERTA} not found — run scripts/score_roberta.py first")
else:
    rb = pd.read_json(ROBERTA, lines=True).set_index("key")["label"]

    def with_roberta(frame):
        out = frame.copy()
        out["roberta"] = (out["type"] + ":" + out["id"].astype(str)).map(rb)
        return out.dropna(subset=["roberta"])

    m3 = with_roberta(uni)
    print(f"  scored: {len(m3):,} of {len(uni):,} framing docs")
    m3["vader3"] = np.select([m3["compound"] < -0.05, m3["compound"] > 0.05],
                             ["negative", "positive"], "neutral")

    def cohen_kappa(a, b):
        po = (a == b).mean()
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(a) | set(b))
        return (po - pe) / (1 - pe)

    print(f"  3-class (neg/neu/pos): agreement={(m3['vader3'] == m3['roberta']).mean()*100:.1f}%"
          f"  kappa={cohen_kappa(m3['vader3'], m3['roberta']):.3f}")
    bin_v, bin_r = m3["vader3"] == "negative", m3["roberta"] == "negative"
    print(f"  binary (negative vs not): agreement={(bin_v == bin_r).mean()*100:.1f}%"
          f"  kappa={cohen_kappa(bin_v, bin_r):.3f}")

    # per-framing negativity under each model (exclusive subset -> tab:model-agreement)
    me = with_roberta(excl)
    me["neg_roberta"] = (me["roberta"] == "negative").astype(int)
    print("  negativity by framing (exclusive subset):")
    for c in RX:
        g = me[me["framing"] == c]
        print(f"    {c:11s} N={len(g):6,}  VADER={g['neg'].mean()*100:4.1f}%"
              f"  RoBERTa={g['neg_roberta'].mean()*100:4.1f}%")
    print("  regression with RoBERTa outcome (same spec):")
    framing_logit(me, "neg_roberta")

# ---------- 2. per-metric sentiment (boundary-aware) ----------
METRICS = {
    "Lines of Code (LOC)":     r"\blines of code\b|\bloc\b",
    "Velocity / Story Points": r"\bvelocity\b|\bstory points?\b|\bsprint points?\b",
    "Throughput / DORA":       r"\bthroughput\b|\bdora\b",
    "KPIs":                    r"\bkpis?\b",
    "OKRs":                    r"\bokrs?\b",
    "Commits / PRs":           r"number of commits?|commit count|pull request count|\bpr count\b",
}
print("\n" + "=" * 60)
print("SENTIMENT PER METRIC (full corpus)")
metric_rows = []
for name, pat in METRICS.items():
    sub = df[df["text"].str.contains(pat, regex=True, na=False)].copy()
    sub["compound"] = vader(sub["text"])
    m, neg, n = stats(sub)
    metric_rows.append({"Métrica": name, "Volume (N)": n,
                        "Média de Sentimento (VADER)": round(m, 3),
                        "Proporção de Negatividade (%)": round(neg * 100, 1)})
    print(f"{name:26s} N={n:6d}  sent={m:+.3f}  neg={neg*100:4.1f}%")
pd.DataFrame(metric_rows).to_csv(_args.metrics_csv, index=False)
print(f"[csv] {_args.metrics_csv}")

# ---------- 3. community profiles (posts) ----------
posts["compound"] = vader(posts["text"])
posts["neg"] = posts["compound"] < -0.05
print("\n" + "=" * 60)
print("COMMUNITY PROFILES (posts)")
SUBS = ["devops", "softwareengineering", "ExperiencedDevs", "cscareerquestions", "womenintech"]
comm = []
for s in SUBS:
    sp = posts[posts["subreddit"] == s]
    mh = (sp["query_group"] == "mental_health").mean()
    comm.append({"sub": s, "n": len(sp), "mh": mh,
                 "sent": sp["compound"].mean(), "neg": sp["neg"].mean()})
    print(f"r/{s:20s} n={len(sp):6d}  mh={mh*100:4.0f}%  "
          f"sent={sp['compound'].mean():+.3f}  neg={sp['neg'].mean()*100:4.1f}%")
comm_df = pd.DataFrame(comm)

# ---------- 4. temporal (posts, 2021-2024) ----------
pst = posts[posts["year"] >= 2021]
yr = pst.groupby(["year", "query_group"]).size().unstack(fill_value=0)
print("\n" + "=" * 60)
print("TEMPORAL (posts by year/group)")
print(yr)
for g in ["mental_health", "metrics_productivity"]:
    if g in yr.columns:
        print(f"{g} growth 2021->2024: {(yr[g][2024]/yr[g][2021]-1)*100:+.1f}%")

# ---------- 5. surveillance / RTO / PIP (posts) ----------
def count_posts(rx):
    return posts["text"].str.contains(rx, regex=True, na=False).sum()
print("\n" + "=" * 60)
print("METRIC-ADJACENT (posts)")
print("surveillance/tracking:", count_posts(r"track|monitor|surveillance|spy"))
print("PIP:", count_posts(r"\bpip\b|performance improvement plan"))
rto = posts[posts["text"].str.contains(r"\brto\b|return to office", regex=True, na=False)]
print("RTO by year:", rto.groupby("year").size().to_dict())

# ---------- 6. catalog (full corpus) ----------
BT = {"sla", "slo", "sli", "kpi", "okr", "pip", "wip", "dora", "mttr", "mttd"}
CATALOG = {
    "DORA (framework)": ["dora metrics", "dora"],
    "DORA - Deployment Frequency": ["deployment frequency", "deploys per day"],
    "DORA - Change Failure Rate": ["change failure rate", "change failure"],
    "DORA - MTTR": ["mttr", "mean time to restore", "mean time to repair"],
    "OKR": ["okr", "objectives and key results"],
    "KPI": ["kpi", "key performance indicator"],
    "Velocity": ["velocity", "sprint velocity"],
    "Story Points": ["story points", "story point"],
    "Cycle Time": ["cycle time"],
    "Throughput": ["throughput"],
    "Code Coverage": ["code coverage", "test coverage"],
    "Lines of Code (LOC)": ["lines of code"],
    "Uptime / Availability": ["uptime", "availability metric"],
    "Performance Review": ["performance review", "perf review", "annual review"],
    "Stack Ranking": ["stack ranking", "stack rank", "forced ranking"],
    "PIP": ["pip", "performance improvement plan"],
    "Technical Debt": ["technical debt", "tech debt"],
    "Developer Productivity": ["developer productivity", "engineering productivity"],
    "SLA / SLO / SLI": ["service level agreement", "service level objective"],
    "Error Budget": ["error budget"],
    "1:1 Cadence": ["1:1", "one-on-one"],
    "Sprint Goal": ["sprint goal", "sprint review"],
}
TYPE = {
    "DORA (framework)": "Framework", "OKR": "Framework",
    "Lines of Code (LOC)": "Metric", "Velocity": "Metric", "Story Points": "Metric",
    "Cycle Time": "Metric", "Throughput": "Metric", "Code Coverage": "Metric",
    "Uptime / Availability": "Metric", "KPI": "Concept", "Technical Debt": "Concept",
    "Developer Productivity": "Concept", "DORA - Deployment Frequency": "Metric",
    "DORA - Change Failure Rate": "Metric", "DORA - MTTR": "Metric",
    "SLA / SLO / SLI": "Metric", "Error Budget": "Metric",
    "Performance Review": "Practice", "Stack Ranking": "Practice", "PIP": "Practice",
    "1:1 Cadence": "Practice", "Sprint Goal": "Practice",
}


def hit(text, term):
    if term in BT:
        return bool(re.search(r"\b" + re.escape(term) + r"\b", text))
    return term in text


# --- (a) regex baseline: the old hardcoded-keyword catalog, kept for validation ---
cat_counts = {}
for metric, terms in CATALOG.items():
    mask = df["text"].apply(lambda t: any(hit(t, x) for x in terms))
    cat_counts[metric] = int(mask.sum())
regex_df = (pd.DataFrame([{"nome": k, "tipo": TYPE.get(k, "Concept"), "posts": v}
                         for k, v in cat_counts.items()])
            .sort_values("posts", ascending=False).reset_index(drop=True))

# --- (b) LLM catalog: open extraction over posts (consolidate_catalog.py), then a manual
# curation pass that drops noise, keeps thesis-relevant items, and adds the `categoria`
# column (MentalHealth / Productivity / Both). The CURATED file is the catalog OF RECORD
# for the figure and the "what is discussed" numbers; the full un-curated LLM catalog is
# loaded separately (step c) only for the coverage validation. Neither is overwritten here
# — rebuild via scripts/llm_catalog.py + scripts/consolidate_catalog.py (+ manual curation).
CURATED_CATALOG = _args.curated_catalog
FULL_LLM_CATALOG = _args.full_catalog


def load_catalog(path):
    d = pd.read_csv(path)
    if "emerging" in d:  # CSV may store TRUE/FALSE as text -> coerce to real bool
        d["emerging"] = (d["emerging"].astype(str).str.strip().str.lower()
                         .isin(("true", "1", "yes")))
    return d


if os.path.exists(CURATED_CATALOG):
    cat_df = load_catalog(CURATED_CATALOG)
    cat_source = "LLM open extraction + manual curation (posts)"
elif os.path.exists(FULL_LLM_CATALOG):
    cat_df = load_catalog(FULL_LLM_CATALOG)
    cat_source = "LLM open extraction, un-curated (posts)"
else:  # fall back to the regex baseline if no LLM catalog has been built yet
    cat_df = regex_df.copy()
    cat_df.insert(0, "rank", cat_df.index + 1)
    cat_df["emerging"] = False
    cat_source = "regex baseline (LLM catalog not found)"
print("\n" + "=" * 60)
print(f"CATALOG — source: {cat_source}")
print(cat_df.head(15).to_string(index=False))

# categoria breakdown (present only in the curated catalog) — the productivity vs
# mental-health split that the thesis hypothesis turns on.
if "categoria" in cat_df:
    print("\nby categoria (distinct items / summed post mentions):")
    for c, g in cat_df.groupby("categoria"):
        print(f"  {c:13s} items={len(g):3d}  posts(sum)={int(g['posts'].sum()):6,}")
    print(f"  {'TOTAL':13s} items={len(cat_df):3d}  "
          f"posts(sum)={int(cat_df['posts'].sum()):6,}")

# --- (c) regex-vs-LLM coverage comparison (validation; see threats section) ---
# Names differ between the two pipelines, so map each regex item to its LLM canonical.
# NB: the regex catalog counts the full corpus while the LLM catalog counts posts, so we
# compare *coverage* (which items each surfaced), not raw magnitudes.
REGEX_TO_LLM = {
    "DORA (framework)": "DORA", "DORA - Deployment Frequency": "Deployment Frequency",
    "DORA - Change Failure Rate": "Change Failure Rate", "DORA - MTTR": "MTTR",
    "OKR": "OKR", "KPI": "KPI", "Velocity": "Velocity", "Story Points": "Story Points",
    "Cycle Time": "Cycle Time", "Throughput": "Throughput", "Code Coverage": "Code Coverage",
    "Lines of Code (LOC)": "Lines of Code", "Uptime / Availability": "Uptime / Availability",
    "Performance Review": "Performance Review", "Stack Ranking": "Stack Ranking", "PIP": "PIP",
    "Technical Debt": "Technical Debt", "Developer Productivity": "Developer Productivity",
    "SLA / SLO / SLI": "SLA / SLO / SLI", "Error Budget": "Error Budget",
    "1:1 Cadence": "1:1 Cadence", "Sprint Goal": "Sprint Goal",
}
# validation is about extraction recall, so it runs against the FULL (un-curated) LLM
# catalog — the curation deliberately drops some baseline items (DORA, OKR, ...), which
# would understate recall if measured against the curated subset.
# If no LLM catalog exists at all we fell back to the regex baseline above; comparing the
# baseline against itself would print a bogus "recall" figure, so skip it instead.
if os.path.exists(FULL_LLM_CATALOG):
    cov_df = load_catalog(FULL_LLM_CATALOG)
else:
    cov_df = None
if cov_df is None:
    print("\n" + "-" * 60)
    print("REGEX vs LLM CATALOG (coverage validation)")
    print(f"  [skip] no LLM catalog at {FULL_LLM_CATALOG} — nothing to validate "
          "against (run scripts/llm_catalog.py + scripts/consolidate_catalog.py)")
    cov_df = cat_df
    llm_names = set()
    recovered, missed = [], []
else:
    llm_names = set(cov_df["nome"])
    recovered = [k for k, v in REGEX_TO_LLM.items() if v in llm_names]
    missed = [k for k, v in REGEX_TO_LLM.items() if v not in llm_names]
emerging = cov_df[cov_df.get("emerging", False) == True] if "emerging" in cov_df else cov_df.iloc[0:0]
print("\n" + "-" * 60)
print("REGEX vs LLM CATALOG (coverage validation)")
print(f"  regex baseline items:           {len(REGEX_TO_LLM)}")
print(f"  recovered by the LLM:           {len(recovered)}/{len(REGEX_TO_LLM)} "
      f"({len(recovered)/len(REGEX_TO_LLM):.0%})")
print(f"  missed by the LLM:              {missed if missed else 'none'}")
emerging_10 = emerging[emerging["posts"] >= 10] if "posts" in emerging else emerging
print(f"  newly discovered (emerging):    {len(emerging)} items "
      f"({len(emerging_10)} with >=10 posts)")
print(f"    top emerging (>=10 posts):    {', '.join(emerging_10.head(10)['nome'])}")

# ---------- 7. metric-MH intersection (posts) ----------
mh_posts = posts[posts["query_group"] == "mental_health"].copy()
mh_posts["mentions_metric"] = mh_posts["text"].str.contains(
    r"\bkpi\b|\bokr\b|\bvelocity\b|story point|performance review|\bpip\b|"
    r"\bscrum\b|\bagile\b|lines of code|\bmetric|tracking", regex=True, na=False)
eng = mh_posts.groupby("mentions_metric")["num_comments"].mean()
print("\nMH posts mean comments by metric-mention:", eng.to_dict())

# =====================================================================
# FIGURES
# =====================================================================
labels = ["System / Pipeline\nMetrics (e.g. DORA)", "Person /\nOutput Metrics",
          "Scrum / Agile\nProcess"]
keys = ["dora", "individual", "scrum"]
colors = [PALETTE["dora"], PALETTE["individual"], PALETTE["scrum"]]
negs = [stats(cat_frames[k])[1] * 100 for k in keys]
sents = [stats(cat_frames[k])[0] for k in keys]
ns = [stats(cat_frames[k])[2] for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
b = axes[0].bar(labels, negs, color=colors, width=.5, edgecolor="white")
for bar, v, n in zip(b, negs, ns):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+.4, f"{v:.1f}%\n(n={n:,})",
                 ha="center", va="bottom", fontsize=8.5)
axes[0].set_ylabel("Negativity Rate (%)"); axes[0].set_ylim(0, max(negs)*1.25)
axes[0].set_title("Proportion of Negative Documents\nby Metric Category")
b = axes[1].bar(labels, sents, color=colors, width=.5, edgecolor="white")
for bar, v in zip(b, sents):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+.005, f"{v:+.3f}",
                 ha="center", va="bottom", fontsize=8.5)
axes[1].axhline(0, color="black", lw=.8, ls="--")
axes[1].set_ylabel("Mean VADER Compound"); axes[1].set_title("Mean Sentiment\nby Metric Category")
fig.suptitle("System/Team vs. Individual Metrics: Community Sentiment", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/metric_category_sentiment.pdf"); plt.close()
print("\n[fig] metric_category_sentiment.pdf")

# community map — two panels: sentiment CONDITIONAL on post type (query_group),
# not over the whole subreddit. Panel A = mental-health posts, Panel B = metrics posts.
def panel_data(group):
    rows = []
    for s in SUBS:
        sp = posts[posts["subreddit"] == s]
        if len(sp) == 0:
            continue
        g = sp[sp["query_group"] == group]
        rows.append({"sub": s, "x": len(g) / len(sp) * 100,
                     "y": g["compound"].mean() if len(g) else 0.0, "n": len(g),
                     "neg": (g["compound"] < -0.05).mean() if len(g) else 0.0})
    return pd.DataFrame(rows)

pa, pb = panel_data("mental_health"), panel_data("metrics_productivity")
print("\nPANEL A — mental-health posts per subreddit:")
print(pa.to_string(index=False))
print("PANEL B — metrics/productivity posts per subreddit:")
print(pb.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
sc = None
for ax, pdat, ttl in [(axes[0], pa, "Mental-Health Posts"),
                      (axes[1], pb, "Productivity / Metrics Posts")]:
    sc = ax.scatter(pdat["x"], pdat["y"], s=pdat["n"] / 15,
                    c=pdat["neg"], cmap="plasma", vmin=.15, vmax=.45,
                    edgecolors="0.25", linewidth=.7, zorder=3)
    ax.margins(x=.22, y=.18)  # headroom so edge labels are not clipped
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    label_bbox = dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.8)
    # Lay out labels in axes-fraction space, modelling each label as a BOX (wide,
    # short) so we can minimise actual label/label and label/bubble overlap rather
    # than point distance. PW/PH ~ panel size in points, for points<->fraction.
    PW, PH = 380.0, 300.0
    fpts = [((r["x"] - x0) / (x1 - x0), (r["y"] - y0) / (y1 - y0))
            for _, r in pdat.iterrows()]
    rads = [(max(r["n"] / 15, 1) / np.pi) ** .5 for _, r in pdat.iterrows()]
    texts = [f"r/{r['sub']}\n(n={int(r['n']):,})" for _, r in pdat.iterrows()]

    def half(text):  # label box half-extents in axes fraction
        w = max(len(l) for l in text.split("\n")) * 4.4
        h = (text.count("\n") + 1) * 9.8
        return w / PW / 2, h / PH / 2

    def overlap(a, b):  # area of intersection of two (cx, cy, hw, hh) boxes
        ox = a[2] + b[2] - abs(a[0] - b[0]); oy = a[3] + b[3] - abs(a[1] - b[1])
        return max(0, ox) * max(0, oy)

    # bubbles are fixed obstacles (boxes sized by marker radius)
    bubbles = [(fx, fy, rads[i] / PW, rads[i] / PH) for i, (fx, fy) in enumerate(fpts)]
    diagonals = [(1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
                 (1, -1, "left", "top"), (-1, -1, "right", "top")]
    # Per point, the 4 candidate label boxes + a fixed cost (bubble overlap +
    # out-of-bounds). Then brute-force the globally best diagonal for every label
    # at once (4 dirs ^ 5 labels), minimising label/label overlap too — greedy
    # placement gets stuck when two bubbles share a y, this does not.
    cand = []
    for i, (fx, fy) in enumerate(fpts):
        rad = rads[i]; hw, hh = half(texts[i])
        ox, oy = (rad * .7 + 6) / PW, (rad + 9) / PH  # anchor offset (fraction)
        opts = []
        for sx, sy, ha, va in diagonals:
            cx, cy = fx + sx * (ox + hw), fy + sy * (oy + hh)  # label box centre
            box = (cx, cy, hw, hh)
            cost = sum(overlap(box, b) for j, b in enumerate(bubbles) if j != i)
            cost += 50 * (max(0, hw - cx) + max(0, cx + hw - 1)     # out of bounds
                          + max(0, hh - cy) + max(0, cy + hh - 1))
            cost += .002 * (sy < 0)  # tiny tie-break: prefer labels above
            opts.append((sx, sy, ha, va, box, cost))
        cand.append(opts)

    best = None
    for combo in itertools.product(*cand):
        pen = sum(c[5] for c in combo)
        for a in range(len(combo)):
            for b in range(a + 1, len(combo)):
                pen += 80 * overlap(combo[a][4], combo[b][4])  # label/label is worst
        if best is None or pen < best[0]:
            best = (pen, combo)

    for i, (sx, sy, ha, va, box, _) in enumerate(best[1]):
        rad = rads[i]
        ax.annotate(texts[i], (pdat.iloc[i]["x"], pdat.iloc[i]["y"]),
                    textcoords="offset points",
                    xytext=(sx * (rad * .7 + 6), sy * (rad + 9)), fontsize=8,
                    ha=ha, va=va, bbox=label_bbox, zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=.6, color="0.55",
                                    shrinkA=1, shrinkB=rad))
    ax.set_xlabel("Share of subreddit's posts on this axis (%)")
    ax.set_title(ttl, fontweight="bold"); ax.grid(True, ls="--", alpha=.4)
axes[0].set_ylabel("Mean VADER Sentiment (this axis only)")
cb = fig.colorbar(sc, ax=axes, fraction=.025, pad=.02); cb.set_label("Negativity Rate")
cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.suptitle("Subreddit Community Map: Sentiment Conditional on Post Type "
             "(bubble size = posts of that type)", fontweight="bold")
fig.savefig(f"{OUT}/subreddit_personality_map.pdf", bbox_inches="tight"); plt.close()
print("[fig] subreddit_personality_map.pdf (two-panel)")

# temporal
monthly = pst.groupby(["month", "query_group"]).size().unstack(fill_value=0).reset_index()
monthly["ms"] = monthly["month"].astype(str)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly["ms"], monthly.get("mental_health", 0), color=PALETTE["mental_health"], lw=2, label="Mental Health")
ax.plot(monthly["ms"], monthly.get("metrics_productivity", 0), color=PALETTE["metrics_productivity"], lw=2, label="Metrics & Productivity")
ax.margins(x=0)
for mth, (lab, yr_) in {"2022-11": ("Mass\nlayoffs begin", .97), "2023-02": ("RTO wave\npeak", .75)}.items():
    if mth in monthly["ms"].values:
        i = monthly[monthly["ms"] == mth].index[0]
        ax.axvline(i, color="gray", ls=":", lw=1.2, alpha=.7)
        ax.text(i+.3, monthly[["mental_health", "metrics_productivity"]].max().max()*yr_, lab, fontsize=7.5, color="gray", va="top")
xt = [i for i, m in enumerate(monthly["ms"]) if m.endswith("-01")]
ax.set_xticks(xt); ax.set_xticklabels([monthly["ms"].iloc[i][:4] for i in xt])
ax.set_xlabel("Year"); ax.set_ylabel("Post Count")
ax.set_title("Monthly Post Volume by Theme (2021-2024)", fontweight="bold")
ax.legend(); ax.grid(True, ls="--", alpha=.3); fig.tight_layout()
fig.savefig(f"{OUT}/temporal_trend_annotated.pdf"); plt.close()
print("[fig] temporal_trend_annotated.pdf")

# catalog figure — curated catalog (LLM open extraction + manual curation), over posts,
# colored by categoria: the productivity vs mental-health split central to the hypothesis.
CAT_TOPN = 25
cat_plot = cat_df.head(CAT_TOPN).sort_values("posts")
if "categoria" in cat_df:
    CMAP = {"MentalHealth": "#c0392b", "Productivity": "#2980b9", "Both": "#8e44ad"}
    CLBL = {"MentalHealth": "Mental Health", "Productivity": "Productivity", "Both": "Both"}
    color_key, xlabel = "categoria", "Posts Mentioning This Item"
    title = f"Catalog of Metrics, Practices, and Well-being Indicators (top {CAT_TOPN})"
else:  # regex fallback colors by tipo
    CMAP = {"Metric": "#2980b9", "Framework": "#27ae60", "Practice": "#e67e22",
            "Concept": "#8e44ad", "Wellbeing": "#c0392b"}
    CLBL = {k: k for k in CMAP}
    color_key, xlabel = "tipo", "Number of Messages Mentioning This Metric/Practice"
    title = f"Catalog of Metrics and Practices (top {CAT_TOPN})"
fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(cat_plot["nome"], cat_plot["posts"],
               color=[CMAP.get(t, "#7f8c8d") for t in cat_plot[color_key]], edgecolor="white")
xmax = max(cat_plot["posts"])
for bar, v in zip(bars, cat_plot["posts"]):
    ax.text(v + xmax*0.005, bar.get_y()+bar.get_height()/2, f"{v:,}", va="center", fontsize=8)
ax.set_xlim(0, xmax * 1.18)  # headroom so the value labels clear the right edge
ax.set_xlabel(xlabel)
ax.set_title(title, fontweight="bold")
# present categories in the legend only when they actually appear in the plotted bars
# (the longest bars sit at the top, so the legend goes lower-right where bars are short)
present = [t for t in CMAP if (cat_plot[color_key] == t).any()]
ax.legend(handles=[mpatches.Patch(color=CMAP[t], label=CLBL[t]) for t in present],
          loc="lower right", fontsize=9, framealpha=.95)
ax.grid(True, axis="x", ls="--", alpha=.4); fig.tight_layout()
fig.savefig(f"{OUT}/metric_catalog.pdf"); plt.close()
print(f"[fig] metric_catalog.pdf (curated/{color_key})")

# surveillance / rto
track_yr = posts[posts["text"].str.contains(r"track|monitor|surveillance|spy", regex=True, na=False)].groupby("year").size()
rto_yr = rto.groupby("year").size()
fig, ax = plt.subplots(figsize=(8, 4))
years = sorted(set(track_yr.index) | set(rto_yr.index))
years = [y for y in years if 2021 <= y <= 2024]
x = np.arange(len(years)); w = .35
ax.bar(x-w/2, [track_yr.get(y, 0) for y in years], w, label="Surveillance / Tracking", color=PALETTE["individual"], edgecolor="white")
ax.bar(x+w/2, [rto_yr.get(y, 0) for y in years], w, label="Return to Office (RTO)", color=PALETTE["scrum"], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(years); ax.set_xlabel("Year"); ax.set_ylabel("Post Count")
ax.set_title("Surveillance & RTO Discussion Volume by Year", fontweight="bold")
ax.legend(); ax.grid(True, axis="y", ls="--", alpha=.4); fig.tight_layout()
fig.savefig(f"{OUT}/surveillance_rto_trends.pdf"); plt.close()
print("[fig] surveillance_rto_trends.pdf")
print("\nAll figures written to", OUT)
