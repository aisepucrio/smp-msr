#!/usr/bin/env python3
"""Analyze the practitioner survey (data/survey_results.csv).

Single source of truth for the survey numbers and figures, mirroring
analyze_corpus.py for the Reddit corpus. Prints every number reported in the
thesis and (re)generates the survey figures into docs/figures/.

The survey is the second, independent leg of the mixed-methods design: where the
corpus observes what developers say unprompted, the survey asks them directly
(RQ1: which metrics/practices are used and how effective they are perceived to
be; RQ2: which measures are desirable but missing). N is small (a convenience
sample), so we report descriptive statistics, not inferential tests.

Run from the repo root:  python scripts/analyze_survey.py
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_ap = argparse.ArgumentParser(description="Analyze the practitioner survey.")
_ap.add_argument("--csv", default="data/survey_results.csv",
                 help="survey export (default: %(default)s)")
_ap.add_argument("--out", default="docs/figures", help="figure output directory")
_args = _ap.parse_args()

CSV = _args.csv
FIGDIR = _args.out

# The survey columns carry the full Portuguese question text. We address them by
# position (stable in the export) and give each a short handle.
COL = {
    "experience": 2, "work_model": 3, "team_size": 4, "company_size": 5,
    "business": 6, "good_metrics": 7, "metrics_used": 8, "ideal_metric": 9,
    "mh_tracking": 10, "mh_practices": 11, "mh_signals": 12, "relatedness": 13,
    "consent": 15, "role": 32,
}

# ---------------------------------------------------------------------------
# Adoption options and their per-option Likert follow-ups.
#
# The pairing is EXPLICIT: (option text, short label, follow-up column index or
# None). It used to be a zip(options, range(...), labels), which silently
# mis-assigned columns whenever the number of options and the number of
# follow-up columns disagreed (9 options vs 8 columns dropped "Other" from the
# productivity table while the adoption figure still plotted all 9 — the figure
# and the table disagreed). "Other" is free text and has no slider, so it is
# None here, and check_columns() below fails loudly if the export does not match.
# ---------------------------------------------------------------------------
PROD_RATED = [
    ("Story points / Velocity (Scrum)", "Story points / velocity", 16),
    ("Deployment frequency (quantas vezes por período é feito deploy)",
     "Deployment frequency", 17),
    ("Change failure rate / MTTR (DORA metrics)", "Change failure / MTTR (DORA)", 18),
    ("Throughput (número de tarefas/PRs entregues por período)", "Throughput", 19),
    ("Code coverage / qualidade de código (ex: SonarQube)", "Code coverage", 20),
    ("Performance review / OKRs individuais", "Performance review / OKRs", 21),
    ("Linhas de código / número de commits", "LOC / commits", 22),
    ("Nenhuma métrica formal", "No formal metric", 23),
    ("Outro", "Other", None),
]
MH_RATED = [
    ("Pulse surveys / pesquisas de clima (ex: Office Vibe, Culture Amp)",
     "Pulse surveys / climate", 24),
    ("eNPS (Employee Net Promoter Score)", "eNPS", 25),
    ("1:1s regulares focados em bem-estar", "Wellbeing 1:1s", 26),
    ("Programa de assistência ao empregado (EAP / psicólogo)",
     "EAP / psychologist", 27),
    ("Benefício de terapia / saúde mental", "Therapy benefit", 28),
    ('"No meeting days" / proteção de tempo de foco', "No-meeting / focus time", 29),
    ("Nenhuma prática formal", "No formal practice", 30),
    ("Outro", "Other", None),
]
PROD_OPTIONS = [o for o, _, _ in PROD_RATED]
PROD_SHORT = [s for _, s, _ in PROD_RATED]
MH_OPTIONS = [o for o, _, _ in MH_RATED]
MH_SHORT = [s for _, s, _ in MH_RATED]


def load():
    if not os.path.exists(CSV):
        raise SystemExit(f"Survey export not found: {CSV}\n"
                         f"Copy the Google Forms export there (or pass --csv).")
    df = pd.read_csv(CSV)
    df.columns = list(range(len(df.columns)))  # address by position
    check_columns(df)
    return df


def check_columns(df):
    """Fail loudly if the export does not match the positional mapping above."""
    ncol = len(df.columns)
    needed = max(list(COL.values())
                 + [c for _, _, c in PROD_RATED + MH_RATED if c is not None])
    if ncol <= needed:
        raise SystemExit(
            f"{CSV} has {ncol} columns but the mapping needs index {needed} "
            f"({needed + 1} columns). The survey was probably reordered or "
            f"shortened — re-check COL / PROD_RATED / MH_RATED before trusting "
            f"any number from this script.")
    mapped = {c for _, _, c in PROD_RATED + MH_RATED if c is not None}
    print(f"survey export: {ncol} columns | positional mapping needs {needed + 1} "
          f"| {len(mapped)} Likert follow-up columns mapped")


def col(df, name):
    return df[COL[name]]


def counts(series, order=None):
    vc = series.value_counts(dropna=False)
    if order:
        vc = vc.reindex(order).dropna()
    return vc


def multiselect_counts(series, options):
    """Count each canonical option as a substring of the cell (comma-safe)."""
    out = {o: 0 for o in options}
    for cell in series.dropna():
        for o in options:
            if o in str(cell):
                out[o] += 1
    return out


def likert(df, series, rated):
    """Per-option (n, mean) for the 1-5 follow-up ratings, counted only for the
    respondents who actually selected the option in the adoption multi-select.

    `rated` is the explicit (option, label, column) list; entries with column None
    (free-text "Other") have no slider and are reported as n/a.

    The survey export contains a few stray ratings: respondents who moved a
    follow-up slider for a metric/practice they did not select as used (e.g. one
    who answered "no formal metric" yet rated throughput). Restricting to selectors
    keeps this table consistent with the adoption figure and with its own scope
    ("rated only by the respondents whose team uses it")."""
    rows = []
    for o, lab, c in rated:
        if c is None:
            rows.append((lab, 0, float("nan")))
            continue
        sel = series.fillna("").str.contains(o, regex=False)
        s = pd.to_numeric(df[c], errors="coerce")[sel].dropna()
        rows.append((lab, len(s), s.mean() if len(s) else float("nan")))
    return rows


def hr(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def report(df):
    n = len(df)
    print(f"\nPractitioner survey — N = {n} responses "
          f"(all consented: {counts(col(df,'consent')).to_dict()})")

    hr("DEMOGRAPHICS")
    for f, order in [
        ("role", None),
        ("experience", ["0-2 anos", "2-5 anos", "5-10 anos", "10+ anos"]),
        ("work_model", ["100% presencial", "Híbrido", "100% remoto"]),
        ("team_size", ["1-5 pessoas", "6-10 pessoas", "10-20 pessoas", "20+ pessoas"]),
        ("company_size", ["Menos de 20 pessoas", "20-100 pessoas",
                          "100-500 pessoas", "500-2.000 pessoas", "Mais de 2.000 pessoas"]),
        ("business", None),
    ]:
        print(f"\n  {f}:")
        for k, v in counts(col(df, f), order).items():
            print(f"    {v:2d} ({v/n*100:4.1f}%)  {k}")

    hr("RQ1 — PRODUCTIVITY METRICS ADOPTED")
    used = multiselect_counts(col(df, "metrics_used"), PROD_OPTIONS)
    for o, s in sorted(zip(PROD_OPTIONS, PROD_SHORT),
                       key=lambda p: -used[p[0]]):
        print(f"    {used[o]:2d} ({used[o]/n*100:4.1f}%)  {s}")
    print("\n  Perceived effectiveness of the metric (1=does not represent, "
          "5=represents very well), among users who selected it:")
    for lab, k, m in likert(df, col(df, "metrics_used"), PROD_RATED):
        print(f"    {lab:28s} n={k:2d}  " + (f"mean={m:.1f}" if k else "n/a"))

    hr("RQ1 — MENTAL-HEALTH TRACKING & PRACTICES")
    print("\n  Does the company track developer well-being?")
    for k, v in counts(col(df, "mh_tracking")).items():
        print(f"    {v:2d} ({v/n*100:4.1f}%)  {k}")
    print("\n  Practices/tools in use:")
    mhu = multiselect_counts(col(df, "mh_practices"), MH_OPTIONS)
    for o, s in sorted(zip(MH_OPTIONS, MH_SHORT), key=lambda p: -mhu[p[0]]):
        print(f"    {mhu[o]:2d} ({mhu[o]/n*100:4.1f}%)  {s}")
    print("\n  Perceived effectiveness of the practice (1=not effective, "
          "5=very effective), among those who selected it:")
    for lab, k, m in likert(df, col(df, "mh_practices"), MH_RATED):
        print(f"    {lab:24s} n={k:2d}  " + (f"mean={m:.1f}" if k else "n/a"))

    hr("CORE HYPOTHESIS — PRODUCTIVITY <-> MENTAL HEALTH ARE RELATED")
    s = pd.to_numeric(col(df, "relatedness"), errors="coerce").dropna()
    print("  (1=completely independent ... 5=inseparable)")
    print(f"  n={len(s)}  mean={s.mean():.2f}  median={s.median():.0f}  "
          f"sd={s.std():.2f}")
    dist = s.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
    for k, v in dist.items():
        print(f"    {k}: {v:2d} ({v/len(s)*100:4.1f}%)  " + "█" * v)
    print(f"  >= 4: {(s>=4).sum()}/{len(s)} ({(s>=4).mean()*100:.0f}%)")
    return dict(n=n, related=s, used=used, mhu=mhu,
                prod_eff=likert(df, col(df, "metrics_used"), PROD_RATED),
                mh_eff=likert(df, col(df, "mh_practices"), MH_RATED))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_relatedness(df):
    s = pd.to_numeric(col(df, "relatedness"), errors="coerce").dropna()
    dist = s.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#f4a259", "#bc4749"]
    bars = ax.bar([1, 2, 3, 4, 5], dist.values, color=colors,
                  edgecolor="0.25", linewidth=.7)
    for b, v in zip(bars, dist.values):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v + .2, str(v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\nCompletely\nindependent", "2", "3", "4",
                        "5\nInseparable"], fontsize=8)
    ax.set_ylabel("Respondents")
    ax.set_ylim(0, max(dist.max(), 1) * 1.18)
    ax.set_title(f"How related are productivity and mental health?  "
                 f"(N={len(s)}, mean={s.mean():.2f})", fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/survey_relatedness.pdf")
    plt.close(fig)


def _adoption_panel(ax, counts_map, options, short, n, title, hl_value):
    pairs = sorted(zip(short, [counts_map[o] for o in options]),
                   key=lambda p: p[1])
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = ["#bc4749" if lab == hl_value else "#4a7ba6" for lab in labels]
    ax.barh(labels, vals, color=colors, edgecolor="0.25", linewidth=.6)
    for i, v in enumerate(vals):
        ax.text(v + .15, i, f"{v}", va="center", fontsize=8)
    ax.set_xlim(0, max(max(vals), 1) * 1.18)
    ax.set_xlabel(f"Respondents (N={n})")
    ax.set_title(title, fontsize=10)
    sns.despine(ax=ax)


def fig_adoption(df, data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _adoption_panel(axes[0], data["used"], PROD_OPTIONS, PROD_SHORT, data["n"],
                    "Productivity metrics used by the team", "No formal metric")
    _adoption_panel(axes[1], data["mhu"], MH_OPTIONS, MH_SHORT, data["n"],
                    "Mental-health practices in place", "No formal practice")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/survey_adoption.pdf")
    plt.close(fig)


def main():
    sns.set_style("whitegrid")
    os.makedirs(FIGDIR, exist_ok=True)
    df = load()
    data = report(df)
    fig_relatedness(df)
    fig_adoption(df, data)
    print(f"\nFigures written to {FIGDIR}/: "
          "survey_relatedness.pdf, survey_adoption.pdf")


if __name__ == "__main__":
    main()
