"""
Generates thesis-quality figures from the collected Reddit dataset.
Run from repo root: python scripts/gerar_figuras_tese.py
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# --- load data ---
csv_files = glob.glob("data/*_posts_unique.csv")
if not csv_files:
    raise SystemExit("No posts CSV found in data/")
df = pd.read_csv(max(csv_files, key=os.path.getctime))
df["year"] = pd.to_datetime(df["created_utc"]).dt.year
df["month"] = pd.to_datetime(df["created_utc"]).dt.to_period("M")

analyzer = SentimentIntensityAnalyzer()

def add_sentiment(frame):
    frame = frame.copy()
    frame["text"] = frame["title"].fillna("") + " " + frame["selftext"].fillna("")
    frame["compound"] = frame["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    frame["negative"] = frame["compound"] < -0.05
    return frame


df = add_sentiment(df)

PALETTE = {
    "mental_health": "#c0392b",
    "metrics_productivity": "#2980b9",
    "dora": "#27ae60",
    "individual": "#e67e22",
    "scrum": "#8e44ad",
    "neutral": "#7f8c8d",
}

# ── Figure 1: DORA vs individual/punitive metrics vs Scrum — negativity & sentiment ──
dora_mask = (
    df["title"].str.contains("dora", case=False, na=False)
    | df["selftext"].str.contains(
        "dora metric|four key|deployment frequency|change failure|mean time to recover",
        case=False, na=False,
    )
)
individual_mask = (
    df["title"].str.contains(
        r"pip\b|performance review|story point|velocity|okr|kpi|lines of code",
        case=False, na=False,
        regex=True,
    )
    | df["selftext"].str.contains(
        "performance improvement plan|individual okr|individual kpi",
        case=False, na=False,
    )
)
scrum_mask = df["title"].str.contains("scrum|agile", case=False, na=False)

groups = {
    "DORA": df[dora_mask],
    "Individual /\nPunitive Metrics": df[individual_mask],
    "Scrum / Agile": df[scrum_mask],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

labels = list(groups.keys())
neg_rates = [g["negative"].mean() * 100 for g in groups.values()]
sentiments = [g["compound"].mean() for g in groups.values()]
colors = [PALETTE["dora"], PALETTE["individual"], PALETTE["scrum"]]
ns = [len(g) for g in groups.values()]

ax = axes[0]
bars = ax.bar(labels, neg_rates, color=colors, width=0.5, edgecolor="white")
for bar, n, v in zip(bars, ns, neg_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%\n(n={n})",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Negativity Rate (%)")
ax.set_title("Proportion of Negative Posts\nby Metric Category")
ax.set_ylim(0, max(neg_rates) * 1.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

ax = axes[1]
bars = ax.bar(labels, sentiments, color=colors, width=0.5, edgecolor="white")
for bar, v in zip(bars, sentiments):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_ylabel("Mean VADER Compound Score")
ax.set_title("Mean Sentiment Score\nby Metric Category")

fig.suptitle("DORA vs. Individual/Punitive Metrics: Community Sentiment", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/metric_category_sentiment.pdf")
plt.close()
print("✓ metric_category_sentiment.pdf")

# ── Figure 2: Subreddit personality matrix ──
SUBS = ["devops", "softwareengineering", "ExperiencedDevs", "cscareerquestions", "womenintech"]
sub_data = []
for sub in SUBS:
    s = df[df["subreddit"] == sub]
    mh_ratio = (s["query_group"] == "mental_health").mean()
    neg_rate = s["negative"].mean()
    mean_sent = s["compound"].mean()
    sub_data.append({"subreddit": sub, "mh_ratio": mh_ratio, "neg_rate": neg_rate,
                     "mean_sent": mean_sent, "n": len(s)})
sub_df = pd.DataFrame(sub_data)

fig, ax = plt.subplots(figsize=(8, 5))
scatter_colors = plt.cm.RdYlGn_r(sub_df["neg_rate"])
sc = ax.scatter(sub_df["mh_ratio"] * 100, sub_df["mean_sent"],
                s=sub_df["n"] / 5, c=sub_df["neg_rate"], cmap="RdYlGn_r",
                vmin=0.15, vmax=0.45, edgecolors="white", linewidth=0.5, zorder=3)
for _, row in sub_df.iterrows():
    ax.annotate(
        f"r/{row['subreddit']}\n(n={row['n']:,})",
        (row["mh_ratio"] * 100, row["mean_sent"]),
        textcoords="offset points", xytext=(10, 4), fontsize=8.5,
    )
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Negativity Rate")
cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Mental Health Post Ratio (%)")
ax.set_ylabel("Mean VADER Sentiment Score")
ax.set_title("Subreddit Personality Map\n(bubble size = post count)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/subreddit_personality_map.pdf")
plt.close()
print("✓ subreddit_personality_map.pdf")

# ── Figure 3: Temporal trend with event annotations ──
# restrict to the 2021-2024 collection window (drops a stray 2020-12 edge post)
monthly = (
    df[df["year"] >= 2021]
    .groupby(["month", "query_group"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
monthly["month_str"] = monthly["month"].astype(str)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly["month_str"], monthly.get("mental_health", 0),
        color=PALETTE["mental_health"], linewidth=2, label="Mental Health")
ax.plot(monthly["month_str"], monthly.get("metrics_productivity", 0),
        color=PALETTE["metrics_productivity"], linewidth=2, label="Metrics & Productivity")
ax.margins(x=0)  # start the axis at the first data point, no left/right padding

# Markers for documented industry events visible in the series.
events = {
    "2022-11": ("Mass\nlayoffs begin", 0.97),
    "2023-02": ("RTO wave\npeak", 0.75),
}
for month, (label, yrel) in events.items():
    if month in monthly["month_str"].values:
        idx = monthly[monthly["month_str"] == month].index[0]
        ymax = monthly[["mental_health", "metrics_productivity"]].max().max()
        ax.axvline(x=idx, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(idx + 0.3, ymax * yrel, label, fontsize=7.5, color="gray",
                va="top", ha="left")

xticks = [i for i, m in enumerate(monthly["month_str"]) if m.endswith("-01")]
ax.set_xticks(xticks)
ax.set_xticklabels([monthly["month_str"].iloc[i][:4] for i in xticks], rotation=0)
ax.set_xlabel("Year")
ax.set_ylabel("Post Count")
ax.set_title("Monthly Post Volume by Theme (2021-2024)", fontweight="bold")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/temporal_trend_annotated.pdf")
plt.close()
print("✓ temporal_trend_annotated.pdf")

# ── Figure 4: Metric catalog — horizontal bar chart ──
catalog = pd.read_csv("data/metricas_catalogadas.csv")
catalog = catalog.sort_values("posts")

# map the Portuguese type labels in the CSV to English for the thesis
TYPE_EN = {"Métrica": "Metric", "Framework": "Framework",
           "Prática": "Practice", "Conceito": "Concept"}
TYPE_COLORS = {
    "Metric": "#2980b9",
    "Framework": "#27ae60",
    "Practice": "#e67e22",
    "Concept": "#8e44ad",
}
colors_cat = [TYPE_COLORS.get(TYPE_EN.get(t, t), "#7f8c8d") for t in catalog["tipo"]]

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(catalog["nome"], catalog["posts"], color=colors_cat, edgecolor="white")
for bar, v in zip(bars, catalog["posts"]):
    ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=8)
ax.set_xlabel("Number of Posts Mentioning This Metric/Practice")
ax.set_title("Metric & Practice Catalog — Reddit Mentions\n(12,740 posts, 2021–2024)", fontweight="bold")
legend_patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/metric_catalog.pdf")
plt.close()
print("✓ metric_catalog.pdf")

# ── Figure 5: womenintech 2024 growth + overall MH ratio by year ──
# r/womenintech shows 0-6 posts/month in 2022-2023, then a clean sustained jump to
# 20-80/month from Jan 2024 — consistent with real growth, not a coverage artifact.
# Pre-2024 data for this subreddit is sparse and should be interpreted cautiously.
wit = df[df["subreddit"] == "womenintech"]
wit_monthly = wit.groupby(["month", "query_group"]).size().unstack(fill_value=0).reset_index()
wit_monthly["month_str"] = wit_monthly["month"].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.stackplot(
    range(len(wit_monthly)),
    wit_monthly.get("mental_health", pd.Series(0, index=wit_monthly.index)),
    wit_monthly.get("metrics_productivity", pd.Series(0, index=wit_monthly.index)),
    labels=["Mental Health", "Metrics & Productivity"],
    colors=[PALETTE["mental_health"], PALETTE["metrics_productivity"]],
    alpha=0.85,
)
xticks = [i for i, m in enumerate(wit_monthly["month_str"]) if m.endswith("-01")]
ax.set_xticks(xticks)
ax.set_xticklabels([wit_monthly["month_str"].iloc[i][:4] for i in xticks])
ax.axvline(x=next(i for i, m in enumerate(wit_monthly["month_str"]) if m == "2024-01"),
           color="gray", linestyle=":", linewidth=1.2)
ax.text(next(i for i, m in enumerate(wit_monthly["month_str"]) if m == "2024-01") + 0.5,
        ax.get_ylim()[1] * 0.85, "2024", color="gray", fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("Posts per Month")
ax.set_title("r/womenintech — Monthly Activity\n(pre-2024 data sparse, interpret cautiously)", fontweight="bold")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
# All subreddits, excl. womenintech for the trend line (it skews 2024 upward)
df_excl = df[df["subreddit"] != "womenintech"]
mh_ratio_excl = (
    df_excl.groupby(["year", "query_group"]).size().unstack(fill_value=0)
    .assign(mh_ratio=lambda x: x["mental_health"] / (x["mental_health"] + x["metrics_productivity"]))
)
years = mh_ratio_excl.index[mh_ratio_excl.index >= 2021]
ax.plot(years, mh_ratio_excl.loc[years, "mh_ratio"] * 100,
        marker="o", color=PALETTE["mental_health"], linewidth=2,
        label="excl. r/womenintech")
ax.set_ylim(60, 90)
ax.set_xticks(years)
ax.set_xlabel("Year")
ax.set_ylabel("Mental Health Post Ratio (%)")
ax.set_title("Mental Health as Share of All Posts\n(4 consistently-covered subreddits)", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(f"{OUT}/mh_ratio_over_time.pdf")
plt.close()
print("✓ mh_ratio_over_time.pdf")

# ── Figure 6: Surveillance / tracking posts over time ──
track_mask = df["title"].str.contains(
    r"track|monitor|surveillance|spy", case=False, na=False, regex=True
) | df["selftext"].str.contains("tracking|monitoring|surveillance", case=False, na=False)

track_yearly = df[track_mask].groupby("year").size()
rto_yearly = df[
    df["title"].str.contains("rto|return to office", case=False, na=False)
].groupby("year").size()

fig, ax = plt.subplots(figsize=(8, 4))
width = 0.35
x = np.arange(len(track_yearly.index))
ax.bar(x - width / 2, track_yearly.values, width, label="Surveillance / Tracking",
       color=PALETTE["individual"], edgecolor="white")
rto_aligned = rto_yearly.reindex(track_yearly.index, fill_value=0)
ax.bar(x + width / 2, rto_aligned.values, width, label="Return to Office (RTO)",
       color=PALETTE["scrum"], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(track_yearly.index)
ax.set_xlabel("Year")
ax.set_ylabel("Post Count")
ax.set_title("Surveillance & RTO Discussion Volume by Year", fontweight="bold")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f"{OUT}/surveillance_rto_trends.pdf")
plt.close()
print("✓ surveillance_rto_trends.pdf")

print(f"\nAll figures saved to {OUT}/")
